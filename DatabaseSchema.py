from sqlalchemy import inspect
import json
from sqlalchemy import create_engine
from constants import ConnectionString

class DatabaseSchema:
    def __init__(self, databaseEngine):
        self.databaseEngine = databaseEngine

    def create_database_schema(self):
        inspector = inspect(self.databaseEngine)
        database_schema = {}

        schemas = inspector.get_schema_names()
        for schema_name in schemas:
            database_schema[schema_name] = {}
            tables = inspector.get_table_names(schema=schema_name)
            if not tables:
                continue
            for table_name in tables:
                # Her tablo için sütun bilgilerini alıyoruz
                primary_keys = inspector.get_pk_constraint(table_name=table_name, schema=schema_name)
                foreign_keys = inspector.get_foreign_keys(table_name=table_name, schema=schema_name)
                columns = inspector.get_columns(table_name=table_name, schema=schema_name)
                
                database_schema[schema_name][table_name] = {
                    "columns": self.get_table_columns_json(columns),
                    "primary_keys": primary_keys["constrained_columns"],
                    "foreign_keys": foreign_keys
                }

        # JSON formatında dosyaya yaz
        with open("database_schema.json", "w") as json_file:
            json.dump(database_schema, json_file, indent=4)

    def get_table_columns_json(self, columns):
        column_info_list = []
        for column in columns:
            column_info = {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column["nullable"],
                "default": column["default"]
            }
            column_info_list.append(column_info)
        return column_info_list

engine = create_engine(ConnectionString)

database = DatabaseSchema(engine)
database.create_database_schema()
