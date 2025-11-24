#!/usr/bin/env bash
set -euo pipefail

# Simple local Postgres bootstrap for crucible_framework.
# Assumes Postgres is running locally and you can connect as the superuser.

DB_USER="crucible_dev"
DB_PASS="crucible_dev_pw"
DB_DEV="crucible_framework_dev"
DB_TEST="crucible_framework_test"
PSQL=${PSQL:-psql}

echo "Creating role and databases (idempotent)..."
$PSQL -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
      CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';
   END IF;
END
\$\$;
SQL

create_db_if_absent() {
  local db="$1"
  if ! $PSQL -tAc "SELECT 1 FROM pg_database WHERE datname='${db}'" | grep -q 1; then
    echo "Creating database ${db}..."
    $PSQL -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${db} OWNER ${DB_USER};"
  else
    echo "Database ${db} already exists."
  fi
}

create_db_if_absent "${DB_DEV}"
create_db_if_absent "${DB_TEST}"

echo "Granting privileges..."
$PSQL -v ON_ERROR_STOP=1 <<SQL
GRANT ALL PRIVILEGES ON DATABASE ${DB_DEV} TO ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${DB_TEST} TO ${DB_USER};
SQL

echo "Running ecto setup for dev..."
MIX_ENV=dev mix ecto.migrate

echo "Running ecto setup for test..."
MIX_ENV=test mix ecto.create
MIX_ENV=test mix ecto.migrate

echo "Done. Credentials are hard-coded in config/dev.exs and config/test.exs."
