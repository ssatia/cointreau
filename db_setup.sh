#!/bin/bash
# Creates required MySQL and InfluxDB databases and associated tables

mysql -u root -pYOUR_MYSQL_PASSWORD < db_scripts/mysql_setup.sql

influx < db_scripts/influx_setup.txt
