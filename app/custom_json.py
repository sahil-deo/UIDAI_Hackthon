from decimal import Decimal
import json


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj)  # or float(obj) if you prefer
        return super().default(obj)