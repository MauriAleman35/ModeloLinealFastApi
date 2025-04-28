import json
from bson import ObjectId
from datetime import datetime, date
from pydantic import BaseModel
from typing import Any

class JSONEncoder(json.JSONEncoder):
    """Encoder personalizado que maneja tipos de MongoDB"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super(JSONEncoder, self).default(obj)

def convert_mongo_objects(obj: Any) -> Any:
    """
    Convierte recursivamente tipos de MongoDB a tipos serializables por JSON
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, list):
        return [convert_mongo_objects(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_mongo_objects(value) for key, value in obj.items()}
    else:
        return obj