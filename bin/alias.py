"""
    Author: Giovanni Montresor
"""

import re

class Alias:
    def __init__(self, alias: str, metrics):
        self.comb = None
        self.metrics = metrics
        self.class_id = metrics[0]
        self.alias = self._validate(self._parse(alias))


    def _parse(self, alias: str):
        if "." in alias:
            self.comb = alias.split(".")
        return alias

    def _validate(self, alias: str):
        cleaned_alias = re.sub(r"[^a-zA-Z]", "", alias)

        validated_alias = ""

        if self.comb:
            validated_alias = "".join(
                [re.sub(r"[^a-zA-Z]", "", comb) for comb in self.comb]
            )
        else:
            validated_alias = cleaned_alias

        if len(validated_alias) == 0:
            raise Exception("Alias non valido")
        else:
            return validated_alias

    def get_class_id(self):
        return self.class_id

    def uinfo(self):
        """
        method to get all the usefull information about the alias.
        : returns: a dictionary with the alias, the class_id and the metrics.
        """
        return {"alias": self.alias, "class_id": self.class_id, "metrics": self.metrics}

    def __str__(self):
        return f"oggetto:\n{self.alias}\n{self.metrics}\nno"


def main():
    prova = Alias("prova", (1, "prova"))
    print(prova)


if __name__ == "__main__":
    main()
