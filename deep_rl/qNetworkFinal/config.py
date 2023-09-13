import os.path
from configparser import ConfigParser
from migration_machine.utils.exceptions import ConfigException

SETTINGS = "settings"
RUNNING_PARAMETERS = "running_parameters"
ALGORITHM_PARAMETERS = "algorithm_parameters"

class Config:
    def __init__(self, config_file):
        self.parameters = None
        self.file = config_file
        self.load_sections()
        self.read_configuration()
        self.load_config()

    def load_sections(self):
        self.sections = [
            SETTINGS,
            RUNNING_PARAMETERS,
            ALGORITHM_PARAMETERS
        ]

    def load_config(self):        
        self.settings = self.get_config(SETTINGS)
        self.running_parameters = self.get_config(RUNNING_PARAMETERS)
        self.algorithm_parameters = self.get_config(ALGORITHM_PARAMETERS)

    def read_configuration(self):
        self.parser = ConfigParser()
        self.parser.read(self.file)
        sections = self.parser.sections()
        self.validate_sections(sections)

    def validate_sections(self, sections):
        difference = list(set(self.sections) - set(sections))
        if len(difference) > 0:
            raise ConfigException(
                "Missing configuration sections {}".format(difference)
            )

    def get_config(self, type):
        return {item[0]: item[1] for item in self.parser.items(type)}
