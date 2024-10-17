import sys

from logbook import Logger, StreamHandler
from numpy import empty
from pandas import DataFrame, read_sql_query, Index, Timedelta, NaT

from zipline.utils.cli import maybe_show_progress

from sqlalchemy import create_engine

handler = StreamHandler(sys.stdout, format_string=" | {record.message}")
logger = Logger(name)
logger.handlers.append(handler)

engine = create_engine('mysql+mysqlconnector://db_user_name:db_password@db_server/db_name')