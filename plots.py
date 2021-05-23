import matplotlib.pyplot as plt

# expr1
steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000]
losses = [12.847401797771454, 8.528288185596466, 8.639106392860413, 7.519662916660309, 7.3442975878715515, 7.194033622741699, 6.486609697341919, 5.743220776319504, 3.041909344494343, 3.122677069157362, 3.0238483054563403, 2.6225296035408974, 2.269424286670983, 0.23272204771637917, 1.4437964661046863, 0.21138997003436089, 0.1028126017190516, 0.0297518476145342, 0.02029488724656403, 0.013779968838207424, 0.010378438048064709, 0.008634462021291256, 0.007238624239107594, 0.6487027060938999, 0.017539329826831818, 0.01932604459580034, 0.017871107382234186, 0.013810200442094356, 0.011172082275152206, 0.009187183866742998, 0.009894682531012222, 0.006791994033847004, 0.004879632106167264, 0.005555690237088129, 0.5678655327646993, 0.009064256941201165, 0.02151937613962218, 0.022160184511449188, 0.01204718864755705, 0.010028489079559222, 0.3158076624968089, 0.044654364057350904, 0.01916524494299665, 0.013009675094508566, 0.0064552352705504745, 0.00951506984711159, 0.006797905749408528, 0.002929171474534087, 0.0027804988640127704, 0.0022678646200802177]
train_accuracies = [0.526, 0.5, 0.5, 0.5, 0.5, 0.599, 0.759, 0.875, 0.96, 0.979, 0.862, 0.972, 0.98, 0.979, 0.997, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999]
test_accuracies = [0.51, 0.5, 0.5, 0.5, 0.5, 0.58, 0.72, 0.88, 0.97, 0.99, 0.83, 0.95, 0.98, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# primes
steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700, 15800, 15900, 16000, 16100, 16200, 16300, 16400, 16500, 16600, 16700, 16800, 16900, 17000, 17100, 17200, 17300, 17400, 17500, 17600, 17700, 17800, 17900, 18000, 18100, 18200, 18300, 18400, 18500, 18600, 18700, 18800, 18900, 19000, 19100, 19200, 19300, 19400, 19500, 19600, 19700, 19800, 19900, 20000, 20100, 20200, 20300, 20400, 20500, 20600, 20700, 20800, 20900, 21000, 21100, 21200, 21300, 21400, 21500, 21600, 21700, 21800, 21900, 22000, 22100, 22200, 22300, 22400, 22500, 22600, 22700, 22800, 22900, 23000, 23100, 23200, 23300, 23400, 23500, 23600, 23700, 23800, 23900, 24000, 24100, 24200, 24300, 24400, 24500, 24600, 24700, 24800, 24900, 25000, 25100, 25200, 25300, 25400, 25500, 25600, 25700, 25800, 25900, 26000, 26100, 26200, 26300, 26400, 26500, 26600, 26700, 26800, 26900, 27000, 27100, 27200, 27300, 27400, 27500, 27600, 27700, 27800, 27900, 28000, 28100, 28200, 28300, 28400, 28500, 28600, 28700, 28800, 28900, 29000, 29100, 29200, 29300, 29400, 29500, 29600, 29700, 29800, 29900, 30000, 30100, 30200, 30300, 30400, 30500, 30600, 30700, 30800, 30900, 31000, 31100, 31200, 31300, 31400, 31500, 31600, 31700, 31800, 31900, 32000, 32100, 32200, 32300, 32400, 32500, 32600, 32700, 32800, 32900, 33000, 33100, 33200, 33300, 33400, 33500, 33600, 33700, 33800, 33900, 34000, 34100, 34200, 34300, 34400, 34500, 34600, 34700, 34800, 34900, 35000, 35100, 35200, 35300, 35400, 35500, 35600, 35700, 35800, 35900]
losses = [9.229631453752518, 7.235661208629608, 4.694188043475151, 5.693979695439339, 4.575481694191694, 5.029872253537178, 4.2922162637114525, 4.5020629316568375, 4.068499222397804, 3.0991772562265396, 4.337986741214991, 4.55441352725029, 4.998121827840805, 4.405517518520355, 3.6210709512233734, 4.773479513823986, 4.041805952787399, 3.378369241952896, 4.221443019807339, 4.030136078596115, 4.404294550418854, 3.7302842289209366, 3.467569373548031, 4.0808184295892715, 3.9999838322401047, 3.8407578617334366, 4.226032733917236, 3.0665111020207405, 4.381819546222687, 3.506277158856392, 4.349161311984062, 4.136723205447197, 4.085723981261253, 4.132634669542313, 3.8490316569805145, 4.219773232936859, 3.683927372097969, 5.622529208660126, 4.3629056215286255, 3.9147678166627884, 3.038051664829254, 3.9515465646982193, 3.324700653553009, 4.098510324954987, 3.248804949223995, 4.636380568146706, 4.250077523291111, 4.485180541872978, 4.726242929697037, 3.825752779841423, 4.5882448852062225, 3.851646840572357, 4.637512177228928, 3.8616125881671906, 3.9529173523187637, 4.297657251358032, 4.010514467954636, 2.9888751208782196, 4.783774957060814, 5.0801993906497955, 3.922593802213669, 4.017382502555847, 3.7734115347266197, 3.847914546728134, 4.036093354225159, 4.7766880095005035, 3.813964694738388, 4.511370375752449, 3.3379669189453125, 4.175074964761734, 2.8724761977791786, 3.8473557084798813, 4.183723919093609, 4.643482983112335, 4.206599697470665, 4.733056217432022, 3.6681832522153854, 3.9585882127285004, 3.4941403716802597, 3.4631430953741074, 4.796638339757919, 4.191558167338371, 4.109088033437729, 3.102487303316593, 5.185450851917267, 3.957007646560669, 3.979870229959488, 4.1455994844436646, 4.650284141302109, 4.05582033097744, 3.821932464838028, 4.079892352223396, 3.364604316651821, 4.531012296676636, 4.650348126888275, 4.680158853530884, 3.659246265888214, 3.880798600614071, 3.9348112642765045, 3.4482601284980774, 3.1722121238708496, 3.2253983952105045, 4.071882762014866, 3.729879468679428, 3.4486272037029266, 4.360250189900398, 3.529010131955147, 4.233190268278122, 3.1148059517145157, 3.3757604137063026, 4.480983719229698, 3.2704034000635147, 3.624462053179741, 3.7960416972637177, 3.8719043359160423, 3.415531501173973, 4.250937655568123, 3.3232230991125107, 4.639030829071999, 4.896358996629715, 4.019531205296516, 3.8523353338241577, 3.8855239152908325, 5.252891927957535, 4.133942402899265, 4.210790455341339, 3.5683213770389557, 4.84961225092411, 3.8289277106523514, 3.791527897119522, 4.531080037355423, 3.67238749563694, 3.6765525490045547, 3.5158337205648422, 3.319801241159439, 3.7556346654891968, 3.999362036585808, 3.6109680235385895, 3.615452393889427, 3.837781351059675, 3.464459791779518, 4.323458179831505, 3.4885463267564774, 3.405907228589058, 3.7787900418043137, 4.0790228098630905, 3.5788723528385162, 4.364148393273354, 4.337404131889343, 4.46526175737381, 4.441725969314575, 3.569196905940771, 2.8465111181139946, 3.2427435740828514, 3.2392648681998253, 3.5558221340179443, 3.432448498904705, 3.5698461532592773, 4.381574034690857, 4.008146330714226, 3.8563350439071655, 4.220136851072311, 4.381897300481796, 3.99696946144104, 4.451658844947815, 3.6127709969878197, 3.7923539131879807, 3.462830498814583, 4.314056947827339, 4.054928660392761, 3.3261841386556625, 4.628174215555191, 3.695950597524643, 4.179523915052414, 3.49780210852623, 3.103121854364872, 4.9732882380485535, 4.1568897515535355, 3.6495148092508316, 3.7224353030323982, 4.621930554509163, 4.137564405798912, 4.1034284979105, 3.4480989575386047, 4.286634527146816, 3.9600502848625183, 3.691094070672989, 3.9772707521915436, 3.665435492992401, 3.4809912592172623, 3.893800988793373, 3.888367623090744, 4.296728223562241, 3.226309932768345, 4.72962561249733, 3.7240081131458282, 4.45275291800499, 5.11524823307991, 4.29102948307991, 4.349093973636627, 4.861921384930611, 3.636491037905216, 3.7233300656080246, 4.296637371182442, 3.267402730882168, 3.313159093260765, 4.13987909257412, 4.822631523013115, 2.9696042239665985, 3.9821779057383537, 3.5144299417734146, 4.727182686328888, 3.013520285487175, 3.5527502670884132, 3.264556422829628, 3.792793497443199, 4.1204996556043625, 4.42097082734108, 3.689759999513626, 3.7903339564800262, 4.181065067648888, 3.8671068251132965, 4.232172310352325, 3.521468922495842, 4.139488264918327, 3.521081790328026, 4.375628590583801, 3.607884295284748, 4.132140010595322, 3.824215605854988, 4.140744090080261, 4.028641611337662, 4.433547541499138, 3.673132836818695, 3.754233628511429, 3.9072361290454865, 3.484719254076481, 4.071371346712112, 4.575055539608002, 4.020945996046066, 3.806500032544136, 3.4800631254911423, 3.5140147507190704, 3.6661966517567635, 4.0457442700862885, 3.6128275990486145, 3.5961344093084335, 4.1008655577898026, 4.437895715236664, 3.5496888160705566, 3.6933925449848175, 3.5756399631500244, 4.092945024371147, 3.8942008167505264, 3.9712701439857483, 4.4129152446985245, 3.433053769171238, 4.2647237330675125, 4.220743477344513, 3.7582752257585526, 4.0487615913152695, 2.961283378303051, 3.928144782781601, 4.406838938593864, 4.159599125385284, 3.3387059569358826, 3.7947460412979126, 3.7695721089839935, 3.5835088044404984, 3.62003093957901, 3.951628528535366, 4.278385356068611, 3.842859297990799, 3.3131965324282646, 3.528694838285446, 4.416733659803867, 3.7571720629930496, 3.4541415125131607, 4.0405867248773575, 4.5080640614032745, 3.3799147605895996, 3.6718143671751022, 4.667972847819328, 3.257596895098686, 3.3831847608089447, 3.962875261902809, 3.864467144012451, 3.940926283597946, 3.926282748579979, 4.937492683529854, 3.4195149913430214, 4.863881379365921, 4.529308885335922, 3.742606684565544, 4.333550572395325, 3.9518546611070633, 3.5165353417396545, 4.010528743267059, 4.466970920562744, 3.7315340638160706, 3.6260454803705215, 4.41127672791481, 4.153662443161011, 3.8427399545907974, 4.36223004758358, 3.4593706130981445, 3.785108596086502, 3.729107104241848, 4.540733590722084, 3.8180556893348694, 3.591354548931122, 3.90858756005764, 4.0619194358587265, 3.59626192599535, 3.7426468431949615, 3.76138449460268, 3.5054435431957245, 3.3127376437187195, 4.255827337503433, 3.6096872836351395, 3.9585234075784683, 3.1208802983164787, 4.085134908556938, 4.802990183234215, 3.866443634033203, 3.679138630628586, 3.4282757863402367, 4.214544303715229, 3.965037003159523, 3.0031545981764793, 4.00088606774807, 2.712601162493229, 4.14095064997673, 3.658815272152424, 3.233927831053734, 3.502767115831375, 4.119616337120533, 3.741890072822571, 3.1263084411621094, 4.990795999765396, 3.862804651260376, 3.4613147377967834, 3.549097955226898, 5.07868567109108, 3.814019590616226, 4.547621160745621, 3.897811621427536, 3.7756025046110153, 3.821631968021393, 4.286423444747925, 4.071992710232735, 3.1748441606760025, 3.89410200715065, 3.8879988491535187, 3.955423265695572, 3.486173450946808, 4.263778179883957, 5.111984923481941, 3.699113130569458]
train_accuracies = [0.7673157162726009, 0.7870653685674548, 0.8013908205841447, 0.8268428372739917, 0.8264255910987482, 0.802086230876217, 0.8272600834492351, 0.8205841446453408, 0.8272600834492351, 0.8272600834492351, 0.8267037552155772, 0.8258692628650904, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8030598052851182, 0.8272600834492351, 0.8269819193324062, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8173852573018081, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.7048678720445063, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8273991655076495, 0.8269819193324062, 0.8272600834492351, 0.8262865090403338, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8273991655076495, 0.8261474269819193, 0.8269819193324062, 0.8269819193324062, 0.8271210013908206, 0.8272600834492351, 0.8272600834492351, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8267037552155772, 0.8272600834492351, 0.8265646731571628, 0.8260083449235048, 0.8257301808066759, 0.8272600834492351, 0.8272600834492351, 0.8269819193324062, 0.8271210013908206, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8273991655076495, 0.8271210013908206, 0.8272600834492351, 0.8272600834492351, 0.8271210013908206, 0.8273991655076495, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8269819193324062, 0.8268428372739917, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.827538247566064, 0.8262865090403338, 0.8269819193324062, 0.8269819193324062, 0.8265646731571628, 0.8273991655076495, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.8272600834492351, 0.827538247566064, 0.827538247566064, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.8273991655076495, 0.827538247566064, 0.8273991655076495, 0.8272600834492351, 0.827538247566064, 0.827538247566064, 0.8212795549374131, 0.8148817802503477, 0.8062586926286509, 0.827538247566064, 0.827538247566064, 0.827538247566064, 0.827538247566064, 0.827538247566064, 0.827538247566064, 0.8273991655076495, 0.8272600834492351, 0.8269819193324062, 0.8268428372739917, 0.8265646731571628, 0.8258692628650904, 0.8243393602225313, 0.8273991655076495, 0.827538247566064]
test_accuracies = [0.762816353017521, 0.7978585334198572, 0.8137573004542504, 0.8325762491888384, 0.8316028552887735, 0.8040233614536015, 0.8345230369889682, 0.8303049967553536, 0.8345230369889682, 0.8345230369889682, 0.8338741077222582, 0.8335496430889033, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8040233614536015, 0.8345230369889682, 0.8335496430889033, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8234912394548994, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.6982478909798832, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8341985723556132, 0.8345230369889682, 0.8332251784555483, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8335496430889033, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8303049967553536, 0.8345230369889682, 0.8338741077222582, 0.8332251784555483, 0.8325762491888384, 0.8345230369889682, 0.8345230369889682, 0.8338741077222582, 0.8341985723556132, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8338741077222582, 0.8338741077222582, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8329007138221933, 0.8329007138221933, 0.8329007138221933, 0.8329007138221933, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8348475016223231, 0.8348475016223231, 0.8348475016223231, 0.8345230369889682, 0.8345230369889682, 0.8293316028552887, 0.8205710577547047, 0.8147306943543153, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8345230369889682, 0.8341985723556132, 0.8341985723556132, 0.8341985723556132, 0.8335496430889033, 0.8319273199221285, 0.8316028552887735, 0.8341985723556132, 0.8345230369889682]

# palindromes
steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500]
losses = [14.053039878606796, 7.3632426261901855, 7.300527513027191, 7.185758650302887, 7.408146381378174, 7.153607428073883, 7.082050383090973, 7.107534468173981, 7.258809208869934, 7.0113659501075745, 7.387986600399017, 7.075717747211456, 7.250105679035187, 7.60592395067215, 7.337092816829681, 7.132554292678833, 7.077276647090912, 7.274044215679169, 7.4576128125190735, 7.111452281475067, 7.101300656795502, 6.869055330753326, 7.382853806018829, 7.163496911525726, 6.956536829471588, 7.09878009557724, 6.9347363114356995, 6.9291921854019165, 7.114713251590729, 7.312127709388733, 7.180887162685394, 6.837097525596619, 7.039754092693329, 7.135663688182831, 6.918524503707886, 7.6931047439575195, 7.412423610687256, 7.035854995250702, 7.316698908805847, 6.998946845531464, 7.1543866991996765, 7.105584263801575, 7.03442770242691, 6.9008601903915405, 7.013597786426544, 7.039095461368561, 6.9530192613601685, 6.885594189167023, 7.009892821311951, 6.927101135253906, 7.081956148147583, 7.289412021636963, 6.993086338043213, 6.932040274143219, 6.898548781871796, 6.886386692523956, 7.339904606342316, 6.9765560030937195, 7.076041162014008, 7.045762896537781, 7.084803283214569, 7.040265321731567, 6.889829874038696, 6.995117485523224, 7.00836718082428, 7.202777445316315, 7.114920556545258, 6.687062323093414, 7.192802667617798, 7.066155195236206, 7.019546449184418, 7.008976757526398, 6.9343708753585815, 7.116189360618591, 6.98397958278656, 6.576996743679047, 7.289514124393463, 7.0415361523628235, 6.935717284679413, 6.971870601177216, 6.842032313346863, 7.08367782831192, 7.084731340408325, 6.754623830318451, 7.267536640167236, 6.974344491958618, 6.977479040622711, 6.969711065292358, 6.892054319381714, 7.031865477561951, 6.644990503787994, 7.744604766368866, 6.966414511203766, 7.142030239105225, 6.769505977630615, 7.092074751853943, 6.972519814968109, 6.980596721172333, 7.402697861194611, 7.011238634586334, 6.8201629519462585, 7.062189996242523, 7.110754549503326, 6.918380320072174, 6.966843128204346]
train_accuracies = [0.49523809523809526, 0.5047619047619047, 0.5042857142857143, 0.5023809523809524, 0.4961904761904762, 0.4980952380952381, 0.5104761904761905, 0.4961904761904762, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.4961904761904762, 0.5047619047619047, 0.4961904761904762, 0.5047619047619047, 0.4942857142857143, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.5047619047619047, 0.49523809523809526, 0.5019047619047619, 0.49523809523809526, 0.5047619047619047, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.4976190476190476, 0.5047619047619047, 0.5047619047619047, 0.49523809523809526, 0.49857142857142855, 0.5047619047619047, 0.49523809523809526, 0.49523809523809526, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.5047619047619047, 0.49666666666666665, 0.49666666666666665, 0.5047619047619047, 0.49666666666666665, 0.5047619047619047, 0.49857142857142855, 0.49523809523809526, 0.49666666666666665, 0.5047619047619047, 0.49857142857142855, 0.49523809523809526, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.49523809523809526, 0.5047619047619047, 0.4961904761904762, 0.5047619047619047, 0.5038095238095238, 0.5028571428571429, 0.5047619047619047, 0.5047619047619047, 0.5028571428571429, 0.49523809523809526, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.5009523809523809, 0.5047619047619047, 0.5047619047619047, 0.5023809523809524, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.49857142857142855, 0.5028571428571429, 0.5047619047619047, 0.5033333333333333, 0.5014285714285714, 0.5023809523809524, 0.49857142857142855, 0.5033333333333333, 0.5033333333333333, 0.5047619047619047, 0.5047619047619047, 0.49952380952380954, 0.4980952380952381, 0.5047619047619047, 0.4980952380952381, 0.5, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.49952380952380954, 0.5047619047619047, 0.49952380952380954, 0.49952380952380954, 0.49952380952380954, 0.5047619047619047, 0.5047619047619047, 0.5047619047619047, 0.5]
test_accuracies = [0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.48777777777777775, 0.5133333333333333, 0.5144444444444445, 0.4822222222222222, 0.5133333333333333, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.5133333333333333, 0.4888888888888889, 0.5133333333333333, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.5111111111111111, 0.5244444444444445, 0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.5133333333333333, 0.4888888888888889, 0.4888888888888889, 0.5111111111111111, 0.5144444444444445, 0.4888888888888889, 0.5111111111111111, 0.5111111111111111, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.5133333333333333, 0.5133333333333333, 0.4888888888888889, 0.5122222222222222, 0.4888888888888889, 0.5144444444444445, 0.5111111111111111, 0.5122222222222222, 0.4888888888888889, 0.5133333333333333, 0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.5111111111111111, 0.4888888888888889, 0.5133333333333333, 0.4888888888888889, 0.5277777777777778, 0.5288888888888889, 0.4888888888888889, 0.4888888888888889, 0.5277777777777778, 0.5111111111111111, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.5166666666666667, 0.4888888888888889, 0.4888888888888889, 0.5211111111111111, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.5144444444444445, 0.5222222222222223, 0.4888888888888889, 0.5188888888888888, 0.5266666666666666, 0.5211111111111111, 0.5144444444444445, 0.5277777777777778, 0.5277777777777778, 0.4888888888888889, 0.4888888888888889, 0.5133333333333333, 0.5222222222222223, 0.4888888888888889, 0.5211111111111111, 0.5188888888888888, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.5133333333333333, 0.4888888888888889, 0.5133333333333333, 0.5133333333333333, 0.5133333333333333, 0.4888888888888889, 0.4888888888888889, 0.4888888888888889, 0.5177777777777778]

plt.title(f'Train Accuracy vs. seen examples')
plt.plot(steps, train_accuracies)
plt.xlabel(f'seen examples')
plt.ylabel(f'Train accuracy')
plt.ylim(0, 1)
plt.show()

plt.title(f'Test Accuracy vs. seen examples')
plt.plot(steps, test_accuracies)
plt.xlabel(f'seen examples')
plt.ylabel(f'Test accuracy')
plt.ylim(0, 1)
plt.show()

plt.title(f'Train loss vs. seen examples')
plt.plot(steps, losses)
plt.xlabel(f'seen examples')
plt.ylabel(f'loss')
plt.show()
