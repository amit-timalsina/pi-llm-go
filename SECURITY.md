# Security policy

## Reporting a vulnerability

Email **amittimalsina21@gmail.com** with a description of the
vulnerability, reproduction steps, and your assessment of the impact.

Please do not open a public GitHub issue for security disclosures.

## Response timeline

- **Acknowledgement**: within 7 days of receipt.
- **Fix**: 14 days for issues affecting API-key handling, RCE, denial-of-service, or anything with CVSS ≥ 7. Other issues addressed on a best-effort basis with a timeline communicated in the acknowledgement.
- **Disclosure**: coordinated. Reporter is credited in the fix's CHANGELOG entry unless they request otherwise.

## Scope

In scope:
- `pi-llm-go` package code, providers, examples.
- Supply chain: `go.mod` dependencies, GitHub Actions workflows.

Out of scope:
- Vulnerabilities in upstream LLM providers (Anthropic, OpenAI, Azure).
  Forward those to the provider directly.
- Vulnerabilities in `github.com/invopop/jsonschema` — report upstream
  (also accepted here if you want a mitigation in our usage).

## Supported versions

Pre-1.0: only the latest minor version receives security fixes. Older
minor versions are not patched; users should upgrade.

Post-1.0: the current and previous minor of the latest major.

## Bounty

No bug bounty at this time. Researcher credit in the release notes is
the standard acknowledgement.
