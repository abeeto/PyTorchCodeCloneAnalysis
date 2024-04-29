from torchtext import data

from utils import get_files_with_data


class MyField(data.RawField):

    def __init__(self):
        super(MyField, self).__init__()

    def preprocess(self, x):
        return int(x)


class WikiSyntheticGeneral(data.Dataset):

    name = 'WikiSyntheticGeneral'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, **kwargs):
        """Create some stuff."""
        import json

        extraction_fields = {
            "text": [("text", text_field)],
            "label": [("label", label_field)],
            "id": [("id", MyField())]
        }

        examples = []
        for file_name in get_files_with_data():
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    entry['id'] = entry['meta']['id']
                    examples.append(data.Example.fromdict(entry, extraction_fields))

        fields = {
            "text": text_field,
            "label": label_field,
            "id": MyField()
        }

        super(WikiSyntheticGeneral, self).__init__(examples, fields, **kwargs)
