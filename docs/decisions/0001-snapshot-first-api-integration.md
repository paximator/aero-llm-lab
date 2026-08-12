# ADR 0001: Snapshot-first external API integration

- Status: accepted
- Date: 2026-08-12

## Decision

External aviation and model services are integrated behind typed interfaces. Every
data-source response is persisted as a content-addressed immutable snapshot before
normalization. Training and evaluation consume manifests of snapshots, never live
API responses.

The production inference path targets an open Mistral model behind the internal
`ModelBackend` contract. Hosted model APIs may later assist annotation or judging,
but their output must carry provider/model/version provenance and cannot define the
frozen test ground truth without human review.

## Consequences

- API provider changes are contained in adapters.
- Experiments remain reproducible when upstream records change.
- Credentials cannot appear in snapshots, manifests, or traces.
- Storage usage increases and snapshot retention must be managed.
- A refresh is an explicit dataset-version event rather than an invisible update.
