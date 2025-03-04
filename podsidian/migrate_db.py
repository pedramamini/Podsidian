#!/usr/bin/env python3
"""
Database migration script for Podsidian.
This script adds new columns to the episodes table for transcript source and URL.
"""

import os
import sqlite3
import click
from pathlib import Path

DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/podsidian/podsidian.db")

def migrate_database(db_path):
    """Add new columns to the episodes table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the columns already exist
    cursor.execute("PRAGMA table_info(episodes)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Add transcript_source column if it doesn't exist
    if 'transcript_source' not in columns:
        click.echo("Adding transcript_source column to episodes table...")
        cursor.execute("ALTER TABLE episodes ADD COLUMN transcript_source VARCHAR(50)")
    
    # Add transcript_url column if it doesn't exist
    if 'transcript_url' not in columns:
        click.echo("Adding transcript_url column to episodes table...")
        cursor.execute("ALTER TABLE episodes ADD COLUMN transcript_url VARCHAR(512)")
    
    # Set default values for existing records
    click.echo("Setting default values for existing records...")
    cursor.execute("UPDATE episodes SET transcript_source = 'whisper' WHERE transcript IS NOT NULL AND transcript_source IS NULL")
    
    conn.commit()
    conn.close()
    click.echo("Database migration completed successfully.")

@click.command()
@click.option('--db-path', default=DEFAULT_DB_PATH, help='Path to the database file')
def main(db_path):
    """Migrate the Podsidian database to add transcript source and URL columns."""
    db_path = os.path.expanduser(db_path)
    
    if not os.path.exists(db_path):
        click.echo(f"Error: Database file not found at {db_path}")
        return
    
    click.echo(f"Migrating database at {db_path}")
    migrate_database(db_path)

if __name__ == '__main__':
    main()
