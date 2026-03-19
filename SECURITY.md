# Security

If you find a security vulnerability, please email security@verra.one or open a private GitHub security advisory. Don't open a public issue.

Include: description, steps to reproduce, potential impact.

## Design

- Parameterized SQL everywhere (no string interpolation)
- OAuth tokens stored with 0600 permissions
- Config written atomically with 0600 permissions
- Data dir created with 0700 permissions
- Deploy inputs validated against allowlists
- Docker binds to 127.0.0.1
- Symlinks rejected during ingestion
- File size limit (500MB)
- Dependencies version-capped
