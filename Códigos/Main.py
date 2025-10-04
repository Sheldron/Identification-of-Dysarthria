from LoadAudios import ConfigAudioFiles
from SaveTrainTest import DefineTestAndTrainDF
from GridSearch import DiscoverBestParams
from ClassificationAlgorithms import ApplyAlgorithms

ConfigAudioFiles()
DefineTestAndTrainDF()
DiscoverBestParams()
ApplyAlgorithms()
