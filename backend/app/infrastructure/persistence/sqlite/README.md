# SQLite Repository

The SQLite implementation is split by responsibility:

- `repository.py` coordinates cache, subscriptions, and durable operations.
- `schema.py` owns table creation, indexes, and additive migrations.
- `codec.py` converts between database rows and public job contracts.

Keep SQL representation details inside this package. Callers import only
`SQLiteJobRepository` from the package root.
