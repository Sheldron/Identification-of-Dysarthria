from LoadAudios import ConfigAudioFiles
from SaveTrainTest import DefineTestAndTrainDF
from GridSearch import DiscoverBestParams
from ClassificationAlgorithms import CtrClassificationAlgorithms

ConfigAudioFiles()
DefineTestAndTrainDF()
DiscoverBestParams()
CtrClassificationAlgorithms()
