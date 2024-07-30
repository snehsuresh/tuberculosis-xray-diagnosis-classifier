#!/bin/sh

# Initialize the Airflow database if it hasn't been initialized
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "Initializing Airflow database..."
    airflow db init
fi

# Create an Airflow user if needed
if ! airflow users list | grep -q "admin"; then
    echo "Creating Airflow admin user..."
    airflow users create -e snehsuresh02@gmail.com -f sneh -l pillai -p admin -r Admin -u admin
fi

# Set PYTHONPATH to include the src directory
export PYTHONPATH="${PYTHONPATH}:/app/airflow/src"

# Start the Airflow scheduler in the background
echo "Starting Airflow scheduler..."
nohup airflow scheduler &

# Start the Airflow webserver
echo "Starting Airflow webserver..."
exec airflow webserver
