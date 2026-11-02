use crate::infer::InferResult;
use crate::tokenization::ValidEncoding;
use std::cmp::max;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use text_embeddings_backend::{BackendError, Batch};
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

/// Queue entry
#[derive(Debug)]
pub struct Entry {
    /// Payload
    pub encoding: ValidEncoding,
    /// Entry metadata
    pub metadata: Metadata,
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    /// InferResponse sender to communicate between the Infer struct and the batching_task
    pub(crate) response_tx: oneshot::Sender<Result<InferResult, BackendError>>,
    /// Tokenization duration
    pub(crate) tokenization: Duration,
    /// Instant when this entry was queued
    pub(crate) queue_time: Instant,
    /// Number of tokens in the prompt
    pub(crate) prompt_tokens: usize,
    /// Pooled embedding
    pub(crate) pooling: bool,
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::Sender<QueueCommand>,
}

impl Queue {
    pub fn new(
        padded_model: bool,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        max_concurrent_requests: usize,
    ) -> Self {
        // Create channels
        // The queue rarely fails to send the QueueCommand due to a lack of buffer size.
        // So, naively increasing the buffer size to twice than `max_concurrent_requests` to prevent the failure temporarily
        let (queue_sender, queue_receiver) = mpsc::channel(2 * max_concurrent_requests);

        // Launch background queue task
        std::thread::spawn(move || {
            queue_blocking_task(
                padded_model,
                max_batch_tokens,
                max_batch_requests,
                max_concurrent_requests,
                queue_receiver,
            )
        });

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .try_send(QueueCommand::Append(Box::new(entry), Span::current()))
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
    }

    /// Append one request's prepared entries without an interleaved NextBatch.
    #[instrument(skip_all)]
    pub fn append_batch(&self, entries: Vec<Entry>) {
        self.queue_sender
            .try_send(QueueCommand::AppendBatch(entries, Span::current()))
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
    }

    /// Get the next batch from the queue
    #[instrument(skip(self))]
    pub async fn next_batch(&self) -> Option<NextBatch> {
        let (response_sender, response_receiver) = oneshot::channel();

        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .try_send(QueueCommand::NextBatch {
                response_sender,
                span: Span::current(),
            })
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect(
            "Queue background task dropped the sender without sending a new batch. This is a bug.",
        )
    }
}

// Background task responsible of the queue state
fn queue_blocking_task(
    padded_model: bool,
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
) {
    let capacity = max_batch_requests.unwrap_or(max_concurrent_requests);

    let mut entries: VecDeque<Entry> = VecDeque::with_capacity(max_concurrent_requests);

    while let Some(cmd) = queue_receiver.blocking_recv() {
        match cmd {
            QueueCommand::Append(entry, span) => {
                let _span = span.entered();
                entries.push_back(*entry);
                let gauge = metrics::gauge!("te_queue_size");
                gauge.increment(1.0);
            }
            QueueCommand::AppendBatch(batch, span) => {
                let _span = span.entered();
                let count = batch.len();
                entries.extend(batch);
                metrics::gauge!("te_queue_size").increment(count as f64);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();

                let mut input_ids = Vec::with_capacity(max_batch_tokens);
                let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
                let mut position_ids = Vec::with_capacity(max_batch_tokens);

                let mut pooled_indices = Vec::with_capacity(capacity);
                let mut raw_indices = Vec::with_capacity(capacity);
                let mut metadata = Vec::with_capacity(capacity);
                let mut cu_seq_lengths = Vec::with_capacity(capacity);
                cu_seq_lengths.push(0);

                let mut current_tokens = 0;
                let mut max_length = 0;

                let mut entry_index = 0;

                while let Some(entry) = entries.pop_front() {
                    // Filter entries where the response receiver was dropped (== entries where the request
                    // was dropped by the client)
                    if entry.metadata.response_tx.is_closed() {
                        let counter = metrics::counter!("te_request_failure", "err" => "dropped");
                        counter.increment(1);
                        continue;
                    }

                    let entry_tokens = entry.encoding.input_ids.len();

                    let total_tokens = if padded_model {
                        (max(max_length, entry_tokens as u32) * (metadata.len() + 1) as u32)
                            as usize
                    } else {
                        current_tokens + entry_tokens
                    };

                    if total_tokens > max_batch_tokens {
                        entries.push_front(entry);
                        break;
                    }

                    match entry.metadata.pooling {
                        true => pooled_indices.push(entry_index),
                        false => raw_indices.push(entry_index),
                    }

                    max_length = max(max_length, entry_tokens as u32);

                    input_ids.extend(entry.encoding.input_ids);
                    token_type_ids.extend(entry.encoding.token_type_ids);
                    position_ids.extend(entry.encoding.position_ids);

                    current_tokens += entry_tokens;
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(current_tokens as u32);

                    entry_index += 1;

                    if Some(metadata.len()) == max_batch_requests {
                        break;
                    }
                }

                let batch_size = metadata.len();
                let next_batch = if metadata.is_empty() {
                    None
                } else {
                    Some((
                        metadata,
                        Batch {
                            input_ids,
                            token_type_ids,
                            position_ids,
                            cumulative_seq_lengths: cu_seq_lengths,
                            max_length,
                            pooled_indices,
                            raw_indices,
                        },
                    ))
                };

                let _ = response_sender.send(next_batch);

                if batch_size > 0 {
                    metrics::histogram!("te_batch_nonempty_size").record(batch_size as f64);
                    metrics::histogram!("te_batch_nonempty_tokens").record(current_tokens as f64);
                } else {
                    metrics::counter!("te_batch_empty_next_count").increment(1);
                }
                let histogram = metrics::histogram!("te_batch_next_size");
                histogram.record(batch_size as f64);
                let histogram = metrics::histogram!("te_batch_next_tokens");
                histogram.record(current_tokens as f64);
                let gauge = metrics::gauge!("te_queue_size");
                gauge.set(entries.len() as f64)
            }
        }
    }
}

pub type NextBatch = (Vec<Metadata>, Batch);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    AppendBatch(Vec<Entry>, Span),
    NextBatch {
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[cfg(test)]
mod request_batch_tests {
    use super::*;

    fn entry(token: u32) -> (Entry, oneshot::Receiver<Result<InferResult, BackendError>>) {
        let (response_tx, response_rx) = oneshot::channel();
        (
            Entry {
                encoding: ValidEncoding {
                    input_ids: vec![token; 3],
                    token_type_ids: vec![0; 3],
                    position_ids: vec![0, 1, 2],
                },
                metadata: Metadata {
                    response_tx,
                    tokenization: Duration::ZERO,
                    queue_time: Instant::now(),
                    prompt_tokens: 3,
                    pooling: true,
                },
            },
            response_rx,
        )
    }

    fn check(
        max_tokens: usize,
        max_requests: Option<usize>,
        cancelled: bool,
        expected: Vec<Vec<u32>>,
    ) {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let queue = Queue::new(false, max_tokens, max_requests, 16);
                let (a, a_rx) = entry(11);
                let (b, b_rx) = entry(22);
                let (c, c_rx) = entry(33);
                let mut live = vec![a_rx, c_rx];
                if cancelled {
                    drop(b_rx);
                } else {
                    live.push(b_rx);
                }
                // This single command must be processed before any NextBatch.
                queue.append_batch(vec![a, b, c]);
                let (d, d_rx) = entry(44);
                live.push(d_rx);
                queue.append(d);
                for tokens in expected {
                    let batch = queue.next_batch().await.unwrap();
                    assert_eq!(
                        batch.1.input_ids,
                        tokens.iter().flat_map(|t| [*t; 3]).collect::<Vec<_>>()
                    );
                    assert_eq!(batch.0.len(), tokens.len());
                    assert_eq!(
                        batch.1.cumulative_seq_lengths,
                        (0..=tokens.len())
                            .map(|i| (i * 3) as u32)
                            .collect::<Vec<_>>()
                    );
                    assert!(batch.1.input_ids.len() <= max_tokens);
                }
                assert!(queue.next_batch().await.is_none());
                drop(live);
            });
    }

    #[test]
    fn atomic_append_preserves_request_and_following_entry_order() {
        check(16, None, false, vec![vec![11, 22, 33, 44]]);
    }
    #[test]
    fn atomic_append_respects_token_budget() {
        check(6, None, false, vec![vec![11, 22], vec![33, 44]]);
    }
    #[test]
    fn atomic_append_respects_request_limit() {
        check(16, Some(2), false, vec![vec![11, 22], vec![33, 44]]);
    }
    #[test]
    fn atomic_append_filters_cancelled_entries() {
        check(16, None, true, vec![vec![11, 33, 44]]);
    }

    #[test]
    fn full_120_candidate_request_fits_exact_token_budget() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let lengths: Vec<usize> =
                    (0..120).map(|i| if i < 30 { 585 } else { 128 }).collect();
                let total: usize = lengths.iter().sum();
                for budget in [total, total - 1] {
                    let queue = Queue::new(false, budget, Some(120), 256);
                    let mut entries = Vec::new();
                    let mut receivers = Vec::new();
                    for (i, &length) in lengths.iter().enumerate() {
                        let (mut value, receiver) = entry(i as u32);
                        value.encoding.input_ids = vec![i as u32; length];
                        value.encoding.token_type_ids = vec![0; length];
                        value.encoding.position_ids = (0..length as u32).collect();
                        value.metadata.prompt_tokens = length;
                        entries.push(value);
                        receivers.push(receiver);
                    }
                    queue.append_batch(entries);
                    let expected_first = if budget == total { 120 } else { 119 };
                    let (metadata, batch) = queue.next_batch().await.unwrap();
                    assert_eq!(metadata.len(), expected_first);
                    assert_eq!(batch.max_length, 585);
                    let mut expected_offset = 0;
                    for (i, &length) in lengths.iter().take(expected_first).enumerate() {
                        assert_eq!(batch.cumulative_seq_lengths[i] as usize, expected_offset);
                        assert_eq!(metadata[i].prompt_tokens, length);
                        assert_eq!(
                            &batch.input_ids[expected_offset..expected_offset + length],
                            vec![i as u32; length]
                        );
                        expected_offset += length;
                    }
                    assert_eq!(batch.input_ids.len(), expected_offset);
                    assert_eq!(
                        batch.cumulative_seq_lengths[expected_first] as usize,
                        expected_offset
                    );
                    if budget < total {
                        let (rest, tail) = queue.next_batch().await.unwrap();
                        assert_eq!(rest.len(), 1);
                        assert_eq!(tail.input_ids, vec![119; 128]);
                        assert_eq!(tail.cumulative_seq_lengths, vec![0, 128]);
                    }
                    assert!(queue.next_batch().await.is_none());
                    drop(receivers);
                }
            });
    }
}
