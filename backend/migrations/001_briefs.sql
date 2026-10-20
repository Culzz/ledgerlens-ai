-- Saved briefs.
--
-- A brief is a list of questions and the programs the model wrote for them. The
-- figures in it are not stored as the browser sent them: they are what the engine
-- and the check returned when the programs were run again on the way in. This
-- table only has to hold the record of that run.
create table if not exists briefs (
    id          text primary key,
    document_id text        not null,
    filename    text        not null,
    rows        jsonb       not null,
    created_at  timestamptz not null default now()
);
