"""Copy the data that was stored before tenant isolation into one person's own schema.

Before isolation, everything lived in one shared schema (public). With isolation on, each signed-in person works in a
schema of their own, so the shared data is no longer visible to anyone. This copies it (it does not move or delete it)
into the schema of the person you name, so they keep what they had.

    uv run python scripts/adopt_legacy_data.py --subject <the person's user id>
    uv run python scripts/adopt_legacy_data.py --email someone@example.com        # looks the id up in the account database
    ... --dry-run                                                                 # only say what would be copied

Safe to repeat: rows that are already there are skipped. To undo: DROP SCHEMA <name> CASCADE (the name is printed).
"""

import argparse
import asyncio
import os
import sys

import psycopg
from psycopg import sql

sys.path.insert(0, os.getcwd())
from app.local import tenancy  # noqa: E402
from app.local.migrations import run_migrations  # noqa: E402

DEFAULT_DB = os.environ.get("VOXSCRIBE_DATABASE_URL", "postgresql://localhost:5432/ideascribe_dev")
DEFAULT_ACCOUNT_DB = "postgresql://localhost:5432/ideascribe_api_dev"

# Parents before children, so every foreign key finds its row.
ORDER = [
    "projects", "features", "labels", "sources", "source_chunks", "source_labels", "meetings", "transcript_segments",
    "transcript_edits", "notes_blocks", "enhanced_blocks", "summaries", "meeting_attachments", "tasks", "ideas",
    "idea_edges", "idea_activity", "idea_prds", "chat_threads", "chat_messages", "calendar_events", "settings",
    "usage", "connector_accounts", "auth_events",
]


async def subject_for_email(email: str, account_db: str) -> str:
    async with await psycopg.AsyncConnection.connect(account_db) as conn:
        rows = await (await conn.execute("SELECT id::text FROM users WHERE lower(email) = lower(%s)", (email,))).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one account for {email}, found {len(rows)}")
    return rows[0][0]


async def main(args) -> None:
    subject = args.subject or await subject_for_email(args.email, args.account_database_url)
    key = tenancy.tenant_key(subject)
    print(f"person {subject}\nschema {key}")
    async with await psycopg.AsyncConnection.connect(args.database_url, autocommit=True) as conn:
        if not args.dry_run:
            await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(key)))
    if not args.dry_run:
        await run_migrations(args.database_url, schema=key)
    async with await psycopg.AsyncConnection.connect(args.database_url) as conn:
        for table in ORDER:
            exists = (await (await conn.execute("SELECT to_regclass(%s)", (f"public.{table}",))).fetchone())[0]
            if not exists:
                continue
            count = (await (await conn.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))).fetchone())[0]
            if args.dry_run:
                print(f"  {table}: {count} row(s) would be copied")
                continue
            cols = [r[0] for r in await (await conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND is_generated = 'NEVER' ORDER BY ordinal_position", (key, table))).fetchall()]  # a generated column fills itself in
            shared = [r[0] for r in await (await conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s", (table,))).fetchall()]
            cols = [c for c in cols if c in shared]
            ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
            await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(key)))
            await conn.execute(sql.SQL("INSERT INTO {t} ({c}) SELECT {c} FROM public.{t} ON CONFLICT DO NOTHING").format(t=sql.Identifier(table), c=ident))
            copied = (await (await conn.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(sql.Identifier(key), sql.Identifier(table)))).fetchone())[0]
            print(f"  {table}: {count} shared row(s), {copied} now in the person's schema")
        await conn.commit()
    print("done" if not args.dry_run else "dry run: nothing was changed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--subject", help="the person's user id (the `sub` of their access token)")
    who.add_argument("--email", help="look the user id up by e-mail in the account database")
    p.add_argument("--database-url", default=DEFAULT_DB)
    p.add_argument("--account-database-url", default=DEFAULT_ACCOUNT_DB)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main(p.parse_args()))
