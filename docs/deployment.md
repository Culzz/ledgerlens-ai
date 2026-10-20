# Deployment

One Vercel project and one optional Postgres. The frontend is a static build on
the CDN; the API is a single Python function at `api/index.py` that imports the
same `create_app()` uvicorn runs locally. There is no second code path and no
CORS between the halves.

## What is where

| Piece | Where | Configured by |
| --- | --- | --- |
| Frontend build | `/` on the deployment | `vercel.json` → build command, `frontend/dist` |
| API | `/api/*` → `api/index.py` | `vercel.json` → rewrites |
| Corpus | `data/finqa/dev.json` | fetched during the build by `scripts/download_data.py` |
| Saved briefs | Postgres | `DATABASE_URL` |
| Answers | model provider over HTTP | `LLM_PROVIDER`, `LLM_MODEL`, `<PROVIDER>_API_KEY` |

The function bundle carries `backend/**` and `data/**`; `frontend/node_modules`
and the build output never enter it. See `includeFiles` in `vercel.json`.

## First deploy

1. Import the repository into Vercel. The framework preset is `null` in
   `vercel.json`, so leave the dashboard's detection alone.
2. Set environment variables (Production and Preview):

   | Name | Value |
   | --- | --- |
   | `MISTRAL_API_KEY` | a free-tier key, or any provider's key with `LLM_PROVIDER` changed |
   | `LLM_PROVIDER` | `mistral` |
   | `LLM_MODEL` | `ministral-14b-2512` |
   | `LLM_MIN_SECONDS_BETWEEN_CALLS` | `2.0` |
   | `DATABASE_URL` | a pooled Postgres URL, if you want share links |

3. Deploy. The build installs the frontend, builds it, fetches the dev split and
   packages the function; the first one takes a few minutes and every push after
   that rebuilds it.

Without a key the app still runs: answers are refused on the card with the
reason, and the rest of the interface works. Without `DATABASE_URL` everything
works except saving a brief, which answers with the reason it cannot.

## Database

One table, one migration file.

```
psql "$DATABASE_URL" -f backend/migrations/001_briefs.sql   # or:
python scripts/migrate.py                                    # records what it applied
python scripts/migrate.py --status
```

`scripts/migrate.py` keeps a `schema_migrations` table and applies each file in
`backend/migrations/` once, inside a transaction with the row that records it.
A failed run leaves nothing half-applied. Migrations are additive; the next one
gets a new number rather than editing this one.

Use the pooled connection string. The function opens a connection per request and
closes it, so a pooler is what keeps a burst of traffic from exhausting Neon's
connection limit.

## Health and logs

`GET /api/health` reports the provider, the model and how many reports loaded.
It does not touch the database; a brief store that is down shows up as a 503 with
a readable message on the share button, not as an unhealthy app.

Function logs are in the Vercel dashboard (Deployments → Functions). Errors are
raised with their type and message; connection strings and keys are never echoed
into a response.

## Limits

- Function duration is 60 seconds (`maxDuration`). A free-tier model call plus one
  repair round is normally ten to thirty; a provider that hangs past the limit
  returns a 504 to the caller.
- The bundle is well under the 500 MB Python limit; the corpus is 30 MB of JSON
  and the dependencies are a handful of wheels.
- Hobby is for non-commercial use. This is a demo, not a product.

## Rollback and recovery

- **Bad deploy:** Vercel keeps every deployment. Promote the previous one from the
  dashboard, or `git revert` and push.
- **Bad migration:** the migration ran in a transaction; revert the file and write
  a new numbered one that undoes it. Nothing in this schema is destructive.
- **Lost database:** briefs are links people saved, not source data. Recreate the
  project, run the migration, and the links stop resolving; nothing else breaks.
- **Spent provider quota:** answers fail with a 503 and the reason. The interface
  and retrieval keep working, so the demo degrades to a reader rather than a 500.
