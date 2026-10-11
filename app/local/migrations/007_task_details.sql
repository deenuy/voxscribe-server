-- Tasks get an owner, a due date, a middle status and an optional project of their own.
-- A task with no project of its own shows its scribe's project (the answer reads COALESCE(task, scribe)).
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS owner TEXT CHECK (owner IS NULL OR length(btrim(owner)) BETWEEN 1 AND 120);
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS due_date DATE;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS project_id TEXT REFERENCES projects(id) ON DELETE SET NULL;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_status_check;
ALTER TABLE tasks ADD CONSTRAINT tasks_status_check CHECK (status IN ('todo', 'in_progress', 'done'));
CREATE INDEX IF NOT EXISTS ix_tasks_due ON tasks (due_date) WHERE due_date IS NOT NULL AND status <> 'done';
CREATE INDEX IF NOT EXISTS ix_tasks_project ON tasks (project_id) WHERE project_id IS NOT NULL;
