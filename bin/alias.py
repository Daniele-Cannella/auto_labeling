"""
    Author: Giovanni Montresor
"""

import re
from bin.log import Log


class Alias:
    """Alias."""

    def __init__(self, alias: str, metrics: tuple):
        """__init__.
        Constructor.
        :param alias: the string with the alias or aliases.
        :type alias: str
        :param metrics: precision and recall values.
        :type metrics: tuple.
        """
        self.comb = None
        self.metrics = metrics
        self.class_id = metrics[0]
        self.alias = self.__validate(self.__parse(alias))

    def __parse(self, alias: str):
        """__parse.
        private method to split the alias if added more than one alias.
        :param alias: alias to split
        :type alias: str
        """
        if "." in alias:
            self.comb = alias.split(".")
        return alias

    def __validate(self, alias: str):
        """__validate.
        private method to remove all the special characters. Raise error if the alias was not valid.
        :param alias: alias to validate.
        :type alias: str
        """
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
        """get_class_id.
        :return: the id of the class.
        """
        return self.class_id

    def get_alias(self):
        """get_alias.
        :return: alias of the object.
        """
        return self.alias

    def get_metrics(self):
        """get_metrics.
        :return: the metrics of the alias.
        """
        return self.metrics

    def uinfo(self):
        """uinfo.
        :return: all the usefull information of the object.
        """
        return {"alias": self.alias, "class_id": self.class_id, "metrics": self.metrics}

    def __str__(self):
        """__str__.
        :return: the string representation of the object.
        """
        return f"oggetto:\n{self.alias}\n{self.metrics}"


def test():
    """main."""
    prova = Alias("prova", (1, "prova"))
    print(prova)


if __name__ == "__main__":
    test()
