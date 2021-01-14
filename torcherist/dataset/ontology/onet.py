import io
import csv

import logging
from cached_property import cached_property
import requests

from collections.abc import MutableMapping, KeysView

from . import CompetencyOntology, Competency, Occupation
from torcherist.config import dataset_dir
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class KeysViewOnlyKeys(KeysView):
    # Subclass that just changes the representation.
    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class Clustering(MutableMapping):
    """ A clustering object acting like a dictionary which key is a cluster concept
    and value is a list of entities associated to that cluster.

    Note:
        Python allows a key to be custom object and not necessarily to be string or
        integer as long as it's hashable, but an object key can be difficult to access.
        `key_transform_fn` is to transform an object key to something else, like string
        or integer for easier accessing clustering.

        `value_item_transfrom_fn` is to convert an abstract entity object into something
        else like string, or integer for further computation. It could be a function to
        concat several attributes of the object.

    Example:
        To create a clustering object that we will iterate through a series of concept and
        entity objects, and build the whole thing, we want to extract the concept name
        attribute as the key and make a tuple of entity's identifier and name as the value.
        ```python
        d = Clustering(
                    name="major_group_competencies_name",
                    key_transform_fn=lambda concept: getattr(concept, "name"),
                    value_item_transform_fn=lambda entity: (getattr(entity, "identifier"), getattr(entity, "name")),
            )
     ```

    Args:
        name (str): Name of the clustering
        key_transform_fn (func): the transform function for keys
        value_item_transform_fn (func): the transform function for values

    """

    def __init__(self,
                 name,
                 key_transform_fn=lambda concept: concept,
                 value_item_transform_fn=lambda entity: entity, ):
        self.name = name
        self.store = dict()
        self.map_raw_key = dict()
        self.map_raw_value = dict()
        self.key_transform_fn = key_transform_fn
        self.value_item_transform_fn = value_item_transform_fn

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = [self.value_item_transform_fn(v) for v in value]
        self.map_raw_value[self.__keytransform__(key)] = value
        self.map_raw_key[self.__keytransform__(key)] = key

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]
        del self.map_raw_key[key]
        del self.map_raw_value[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return self.key_transform_fn(key)

    def keys(self):
        return KeysViewOnlyKeys(self)

    def raw_items(self):
        return self.map_raw_value.items()


majorgroupname = {
    '11': 'Management',
    '13': 'Business and Financial Operations',
    '15': 'Computer and Mathematical',
    '17': 'Architecture and Engineering',
    '19': 'Life, Physical, and Social Science',
    '21': 'Community and Social Service',
    '23': 'Legal',
    '25': 'Education, Training, and Library',
    '27': 'Arts, Design, Entertainment, Sports, and Media',
    '29': 'Healthcare Practitioners and Technical',
    '31': 'Healthcare Support',
    '33': 'Protective Service',
    '35': 'Food Preparation and Serving Related',
    '37': 'Building and Grounds Cleaning and Maintenance',
    '39': 'Personal Care and Service',
    '41': 'Sales and Related',
    '43': 'Office and Administrative Support',
    '45': 'Farming, Fishing, and Forestry',
    '47': 'Construction and Extraction',
    '49': 'Installation, Maintenance, and Repair',
    '51': 'Production',
    '53': 'Transportation and Material Moving',
    '55': 'Military Specific'
}


class OnetReader:
    """
    An object that downloads files from the ONET site
    """

    def __init__(self, onet_dir: Path, version="25_1"):
        self.onet_dir = onet_dir
        self.onet_dir.mkdir(exist_ok=True)
        self.downloader = OnetDownloader(to=self.onet_dir, version=version)

    def reader(self, filename: str) -> csv.DictReader:
        """
        Ensures that the given ONET data file is present, either by
        using a cached copy or downloading from S3

        Args:
            filename: unpathed filename of an ONET file (eg. Skills, Tools)

        Returns:
            csv.DictReader
        """
        target_file = self.onet_dir / filename
        if not target_file.exists():
            # download & save to onet_dir/filename
            self.downloader.download(filename)
        # now onet_dir/filename exists
        return csv.DictReader(target_file.open("r"), delimiter="\t")


class OnetDownloader(object):
    """
    Downloads `version` of ONET
     and save to `to` dir
    """

    def __init__(self, to: Path, version: str):
        self.to = to
        self.to.mkdir(exist_ok=True, parents=True)
        self.version = version
        self.url_prefix = f'http://www.onetcenter.org/dl_files/database/db_{version}_text'

    def download(self, source_file):
        url = f'{self.url_prefix}/{source_file}.txt'
        response = requests.get(url)
        with open(self.to / source_file, "w") as f:
            f.write(response.text)


class Onet(CompetencyOntology):
    def __init__(self, version="25_1", onet_dir=None):
        super().__init__()
        self.reader = OnetReader(onet_dir=onet_dir or Path(dataset_dir) / "onet",
                                 version=version)
        self.name = 'onet'
        self.competency_framework.name = 'onet_ksat'
        self.competency_framework.description = 'ONET Knowledge, Skills, Abilities, Tools, and Technology'
        self._build()

    def _build(self):
        reader = self.reader
        description_lookup = {}
        logging.info('Processing Content Model Reference')
        for row in reader.reader('Content Model Reference'):
            description_lookup[row['Element ID']] = row['Description']

        logging.info('Processing occupation data')
        for row in reader.reader('Occupation Data'):
            occupation = Occupation(
                identifier=row['O*NET-SOC Code'],
                name=row['Title'],
                description=row['Description'],
                categories=['O*NET-SOC Occupation'],
            )
            major_group_num = row['O*NET-SOC Code'][0:2]
            major_group = Occupation(
                identifier=major_group_num,
                name=majorgroupname[major_group_num],
                categories=['O*NET-SOC Major Group']
            )
            occupation.add_parent(major_group)
            self.add_occupation(occupation)
            self.add_occupation(major_group)

        for content_model_file in [
            'Knowledge', 'Abilities', 'Skills', 'Interests',
            'Work Styles', 'Work Context', 'Work Activities',
        ]:
            logging.info(f"Processing {content_model_file}")
            for row in reader.reader(content_model_file):
                scale, score = row["Scale ID"], float(row["Data Value"])
                # for knowledge, AbilitSkills, WorkStyle, WorkActivity: importance IM (1-5)
                # for Interests: RIASEC level OI of interest (1-7)
                # for Work Context: percent of respondents endorsing CXP (0-100)
                if scale in ['IM', 'OI', 'CXP']:
                    competency = Competency(
                        identifier=row['Element ID'],
                        name=row['Element Name'],
                        categories=[content_model_file, f"{scale}-{score}"],
                        competencyText=description_lookup[row['Element ID']]
                    )
                    self.add_competency(competency)
                    occupation = Occupation(identifier=row['O*NET-SOC Code'])
                    self.add_edge(competency=competency, occupation=occupation)

        logging.info('Processing tools and technology')
        for file in ["Technology Skills", "Tools Used"]:
            for row in reader.reader(file):
                commodity = row['Commodity Title']
                cat = "Tech Skills" if file == "Technology Skills" else "Tools"
                hot = ["HOT"] if file == "Technology Skills" and row["Hot Technology"] == "Y" else []
                # no competencyText
                competency = Competency(
                    # id = <commodity id>-<skill name> since otherwise only add commodity
                    identifier=row["Commodity Code"] + "-" + row["Example"],
                    name=row['Example'],
                    categories=[cat] + hot + [commodity],
                )
                self.add_competency(competency)
                occupation = Occupation(identifier=row['O*NET-SOC Code'])
                self.add_edge(competency=competency, occupation=occupation)

        logging.info("Processing tasks")
        for row in reader.reader("Task Statements"):
            competency = Competency(
                identifier=row['Task ID'],
                name=row['Task'],
                categories=["Tasks"] + [row["Task Type"]],
            )
            self.add_competency(competency)
            occupation = Occupation(identifier=row['O*NET-SOC Code'])
            self.add_edge(competency=competency, occupation=occupation)

    @cached_property
    def all_soc(self):
        occupations = self.occupations
        soc = []
        for occ in occupations:
            if 'O*NET-SOC Occupation' in occ.other_attributes['categories']:
                soc.append(occ.identifier)
        return sorted(soc)

    @cached_property
    def all_major_groups(self):
        occupations = self.occupations
        major_groups = []
        for occ in occupations:
            if 'O*NET-SOC Major Group' in occ.other_attributes['categories']:
                major_groups.append(occ)
        return sorted(major_groups, key=lambda k: k.identifier)

    @cached_property
    def all_major_groups_occ(self):
        occ = self.filter_by(lambda edge: len(edge.occupation.identifier) == 2)
        return occ.occupations

    @cached_property
    def competency_categories(self):
        return set(c.categories[0] for c in self.competencies)

    @cached_property
    def major_group_occupation_name_clustering(self):
        d = Clustering(
            name="major_group_occupations_name",
            key_transform_fn=lambda concept: getattr(concept, "name"),
            value_item_transform_fn=lambda entity: (getattr(entity, "identifier"), getattr(entity, "name")),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = [child for child in mg.children]
        return d

    @cached_property
    def major_group_occupation_description_clustering(self):
        d = Clustering(
            name="major_group_occupations_description",
            key_transform_fn=lambda concept: getattr(concept, "name"),
            value_item_transform_fn=lambda entity: (
                getattr(entity, "identifier"),
                ' '.join([str(getattr(entity, "name")), str(getattr(entity, "other_attributes").get("description"))])),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = [child for child in mg.children]
        return d

    @cached_property
    def major_group_competencies_name_clustering(self):
        d = Clustering(
            name="major_group_competencies_name",
            key_transform_fn=lambda concept: getattr(concept, "name"),
            value_item_transform_fn=lambda entity: (getattr(entity, "identifier"), getattr(entity, "name")),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = self.filter_by(lambda edge: edge.occupation.identifier[:2] == mg.identifier[:2]).competencies
        return d

    @cached_property
    def major_group_competencies_description_clustering(self):
        d = Clustering(
            name="major_group_competencies_description",
            key_transform_fn=lambda concept: getattr(concept, "name"),
            value_item_transform_fn=lambda entity: (
                getattr(entity, "identifier"),
                ' '.join(
                    [str(getattr(entity, "name")), str(getattr(entity, "other_attributes").get("competencyText"))])),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = self.filter_by(lambda edge: edge.occupation.identifier[:2] == mg.identifier[:2]).competencies
        return d

    def generate_clusterings(self):
        return [
            self.major_group_occupation_name_clustering,
            self.major_group_occupation_description_clustering,
            self.major_group_competencies_name_clustering,
            self.major_group_competencies_description_clustering
        ]
