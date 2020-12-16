from torcher.dataset.ontology.onet import Onet

import logging
import datetime


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)8.8s] %(message)s",
#     handlers=[logging.StreamHandler(),
#               logging.FileHandler(f'log/{datetime.datetime.now().isoformat().replace(":", "-")}.log', encoding='utf-8')],
# )
logging.getLogger().setLevel(logging.INFO)

onet = Onet()