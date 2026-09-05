-- ============================================================
-- CurricAgents — Supabase Schema (safe to re-run)
-- Paste this into Supabase SQL Editor and click RUN
-- All statements are idempotent — safe to run multiple times
-- ============================================================

-- ── Tables ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS learner_state (
    student_id   TEXT        NOT NULL,
    domain       TEXT        NOT NULL,
    state_json   TEXT        NOT NULL,
    updated_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (student_id, domain)
);

CREATE TABLE IF NOT EXISTS pilot_scores (
    id           SERIAL PRIMARY KEY,
    student_id   TEXT        NOT NULL,
    domain       TEXT        NOT NULL,
    test_type    TEXT        NOT NULL,
    score_pct    REAL        NOT NULL,
    n_correct    INTEGER,
    n_total      INTEGER,
    group_label  TEXT        NOT NULL DEFAULT 'experimental',
    timestamp    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (student_id, domain, test_type)
);

CREATE TABLE IF NOT EXISTS assessment_log (
    id               SERIAL PRIMARY KEY,
    student_id       TEXT        NOT NULL,
    domain           TEXT        NOT NULL,
    topic            TEXT        NOT NULL,
    item_type        TEXT        NOT NULL,
    item_text        TEXT,
    response         TEXT,
    grade            REAL,
    justification    TEXT,
    raw_llm_response TEXT,
    graded_by        TEXT        NOT NULL DEFAULT 'exact_match',
    flagged          BOOLEAN     NOT NULL DEFAULT FALSE,
    latency_ms       INTEGER,
    timestamp        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS survey_responses (
    id           SERIAL PRIMARY KEY,
    student_id   TEXT        NOT NULL,
    domain       TEXT        NOT NULL,
    group_label  TEXT        NOT NULL DEFAULT 'experimental',
    q1_ease      INTEGER,
    q2_helpful   INTEGER,
    q3_adaptive  INTEGER,
    q4_recommend INTEGER,
    q5_prefer    INTEGER,
    comments     TEXT,
    timestamp    TIMESTAMPTZ DEFAULT now()
);

-- ── Row Level Security ────────────────────────────────────────

ALTER TABLE learner_state    ENABLE ROW LEVEL SECURITY;
ALTER TABLE pilot_scores     ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessment_log   ENABLE ROW LEVEL SECURITY;
ALTER TABLE survey_responses ENABLE ROW LEVEL SECURITY;

-- Drop existing policies first so re-runs don't fail
DROP POLICY IF EXISTS "allow_all_learner_state"  ON learner_state;
DROP POLICY IF EXISTS "allow_all_pilot_scores"   ON pilot_scores;
DROP POLICY IF EXISTS "allow_all_assessment_log" ON assessment_log;
DROP POLICY IF EXISTS "allow_all_survey"         ON survey_responses;

-- Recreate policies
CREATE POLICY "allow_all_learner_state"  ON learner_state     FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_pilot_scores"   ON pilot_scores      FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_assessment_log" ON assessment_log    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_survey"         ON survey_responses  FOR ALL USING (true) WITH CHECK (true);
