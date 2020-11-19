
clear all
close all
clc

rng('default')

%[filename pathname] = uigetfile({'*'},'File Selector'); %load the supplemental file with zircon age eHfT data

% Read in data, format is name header and two columns of info, for our example we use age + Hf, but any 2D data will work
[numbers text1, data] = xlsread('/Users/kurtsundell/Desktop/Manuscript_v01/sensitivity_input1.xlsx');

AgeHf1 = numbers(:,1:2);
AgeHf2 = numbers(:,3:4);

n = length(AgeHf1);

xmin = 0;
xmax = 4400;
xint = 1; %interval for KDE
x = xmin:xint:xmax;
ymin = -41;
ymax = 20;
bandwidth_x = 20; % kernel bandwidth x (Myr)
bandwidth_y = 1; % kernel bandwidth y (Epsilon units)
gridspc = 2^9; % resolution of bivariae KDEs (how many pixels), has to be in powers of 2, no need to go over go over 2^12

interval = 100;

for j = 1:15
	
	for k = 1:20
		
		F = datasample(AgeHf1,interval*j,'Replace',true);
		G = datasample(AgeHf1,interval*j,'Replace',true);
		
		kde1 = kde(F(:,1),bandwidth_x.*ones(size(F(:,1))),xmin,xmax,xint); %make KDE
		kde2 = kde(G(:,1),bandwidth_x.*ones(size(G(:,1))),xmin,xmax,xint); %make KDE
		
		cdf1 = [0;(1:(n*2))'/(n*2);1];
		cdf2 = [0;(1:(n*2))'/(n*2);1];
		
		MIN_XY=[xmin,ymin]; %grid params for bivariate KDEs
		MAX_XY=[xmax,ymax]; %grid params for bivariate KDEs
		
		% viridis colormap that clips 0 values at 95%
		%cmap = [1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;0.280893580000000,0.0789070300000000,0.402329440000000;0.281445810000000,0.0843197000000000,0.407414040000000;0.281923580000000,0.0896662200000000,0.412415210000000;0.282327390000000,0.0949554500000000,0.417330860000000;0.282656330000000,0.100195760000000,0.422160320000000;0.282910490000000,0.105393450000000,0.426902020000000;0.283090950000000,0.110553070000000,0.431553750000000;0.283197040000000,0.115679660000000,0.436114820000000;0.283228820000000,0.120777010000000,0.440584040000000;0.283186840000000,0.125847990000000,0.444960000000000;0.283072000000000,0.130894770000000,0.449241270000000;0.282883890000000,0.135920050000000,0.453427340000000;0.282622970000000,0.140925560000000,0.457517260000000;0.282290370000000,0.145912330000000,0.461509950000000;0.281886760000000,0.150881470000000,0.465404740000000;0.281412280000000,0.155834250000000,0.469201280000000;0.280867730000000,0.160771320000000,0.472899090000000;0.280254680000000,0.165692720000000,0.476497620000000;0.279573990000000,0.170598840000000,0.479996750000000;0.278826180000000,0.175490200000000,0.483396540000000;0.278012360000000,0.180366840000000,0.486697020000000;0.277134370000000,0.185228360000000,0.489898310000000;0.276193760000000,0.190074470000000,0.493000740000000;0.275191160000000,0.194905400000000,0.496004880000000;0.274128020000000,0.199720860000000,0.498911310000000;0.273005960000000,0.204520490000000,0.501720760000000;0.271828120000000,0.209303060000000,0.504434130000000;0.270594730000000,0.214068990000000,0.507052430000000;0.269307560000000,0.218817820000000,0.509576780000000;0.267968460000000,0.223549110000000,0.512008400000000;0.266579840000000,0.228262100000000,0.514348700000000;0.265144500000000,0.232955930000000,0.516599300000000;0.263663200000000,0.237630780000000,0.518761630000000;0.262138010000000,0.242286190000000,0.520837360000000;0.260571030000000,0.246921700000000,0.522828220000000;0.258964510000000,0.251536850000000,0.524736090000000;0.257322440000000,0.256130400000000,0.526563320000000;0.255645190000000,0.260702840000000,0.528311520000000;0.253934980000000,0.265253840000000,0.529982730000000;0.252194040000000,0.269783060000000,0.531579050000000;0.250424620000000,0.274290240000000,0.533102610000000;0.248628990000000,0.278775090000000,0.534555610000000;0.246811400000000,0.283236620000000,0.535940930000000;0.244972080000000,0.287675470000000,0.537260180000000;0.243113240000000,0.292091540000000,0.538515610000000;0.241237080000000,0.296484710000000,0.539709460000000;0.239345750000000,0.300854940000000,0.540843980000000;0.237441380000000,0.305202220000000,0.541921400000000;0.235526060000000,0.309526570000000,0.542943960000000;0.233602770000000,0.313827730000000,0.543914240000000;0.231673500000000,0.318105800000000,0.544834440000000;0.229739260000000,0.322361270000000,0.545706330000000;0.227801920000000,0.326594320000000,0.546532000000000;0.225863300000000,0.330805150000000,0.547313530000000;0.223925150000000,0.334994000000000,0.548052910000000;0.221989150000000,0.339161140000000,0.548752110000000;0.220056910000000,0.343306880000000,0.549413040000000;0.218129950000000,0.347431540000000,0.550037550000000;0.216209710000000,0.351535480000000,0.550627430000000;0.214297570000000,0.355619070000000,0.551184400000000;0.212394770000000,0.359682730000000,0.551710110000000;0.210503100000000,0.363726710000000,0.552206460000000;0.208623420000000,0.367751510000000,0.552674860000000;0.206756280000000,0.371757750000000,0.553116530000000;0.204902570000000,0.375745890000000,0.553532820000000;0.203063090000000,0.379716440000000,0.553925050000000;0.201238540000000,0.383669890000000,0.554294410000000;0.199429500000000,0.387606780000000,0.554642050000000;0.197636500000000,0.391527620000000,0.554969050000000;0.195859930000000,0.395432970000000,0.555276370000000;0.194100090000000,0.399323360000000,0.555564940000000;0.192357190000000,0.403199340000000,0.555835590000000;0.190631350000000,0.407061480000000,0.556089070000000;0.188922590000000,0.410910330000000,0.556326060000000;0.187230830000000,0.414746450000000,0.556547170000000;0.185555930000000,0.418570400000000,0.556752920000000;0.183897630000000,0.422382750000000,0.556943770000000;0.182255610000000,0.426184050000000,0.557120100000000;0.180629490000000,0.429974860000000,0.557282210000000;0.179018790000000,0.433755720000000,0.557430350000000;0.177422980000000,0.437527200000000,0.557564660000000;0.175841480000000,0.441289810000000,0.557685260000000;0.174273630000000,0.445044100000000,0.557792160000000;0.172718760000000,0.448790600000000,0.557885320000000;0.171176150000000,0.452529800000000,0.557964640000000;0.169645730000000,0.456262090000000,0.558030340000000;0.168126410000000,0.459988020000000,0.558081990000000;0.166617100000000,0.463708130000000,0.558119130000000;0.165117030000000,0.467422900000000,0.558141410000000;0.163625430000000,0.471132780000000,0.558148420000000;0.162141550000000,0.474838210000000,0.558139670000000;0.160664670000000,0.478539610000000,0.558114660000000;0.159194130000000,0.482237400000000,0.558072800000000;0.157729330000000,0.485931970000000,0.558013470000000;0.156269730000000,0.489623700000000,0.557936000000000;0.154814880000000,0.493312930000000,0.557839670000000;0.153364450000000,0.497000030000000,0.557723710000000;0.151918200000000,0.500685290000000,0.557587330000000;0.150476050000000,0.504369040000000,0.557429680000000;0.149039180000000,0.508051360000000,0.557250500000000;0.147607310000000,0.511732630000000,0.557048610000000;0.146180260000000,0.515413160000000,0.556822710000000;0.144758630000000,0.519093190000000,0.556571810000000;0.143343270000000,0.522772920000000,0.556294910000000;0.141935270000000,0.526452540000000,0.555990970000000;0.140535990000000,0.530132190000000,0.555658930000000;0.139147080000000,0.533812010000000,0.555297730000000;0.137770480000000,0.537492130000000,0.554906250000000;0.136408500000000,0.541172640000000,0.554483390000000;0.135065610000000,0.544853350000000,0.554029060000000;0.133742990000000,0.548534580000000,0.553541080000000;0.132444010000000,0.552216370000000,0.553018280000000;0.131172490000000,0.555898720000000,0.552459480000000;0.129932700000000,0.559581620000000,0.551863540000000;0.128729380000000,0.563265030000000,0.551229270000000;0.127567710000000,0.566948910000000,0.550555510000000;0.126453380000000,0.570633160000000,0.549841100000000;0.125393830000000,0.574317540000000,0.549085640000000;0.124394740000000,0.578002050000000,0.548287400000000;0.123462810000000,0.581686610000000,0.547444980000000;0.122605620000000,0.585371050000000,0.546557220000000;0.121831220000000,0.589055210000000,0.545622980000000;0.121148070000000,0.592738890000000,0.544641140000000;0.120565010000000,0.596421870000000,0.543610580000000;0.120091540000000,0.600103870000000,0.542530430000000;0.119737560000000,0.603784590000000,0.541399990000000;0.119511630000000,0.607463880000000,0.540217510000000;0.119423410000000,0.611141460000000,0.538981920000000;0.119482550000000,0.614817020000000,0.537692190000000;0.119698580000000,0.618490250000000,0.536347330000000;0.120080790000000,0.622160810000000,0.534946330000000;0.120638240000000,0.625828330000000,0.533488340000000;0.121379720000000,0.629492420000000,0.531972750000000;0.122312440000000,0.633152770000000,0.530398080000000;0.123443580000000,0.636808990000000,0.528763430000000;0.124779530000000,0.640460690000000,0.527067920000000;0.126325810000000,0.644107440000000,0.525310690000000;0.128087030000000,0.647748810000000,0.523490920000000;0.130066880000000,0.651384360000000,0.521607910000000;0.132267970000000,0.655013630000000,0.519660860000000;0.134691830000000,0.658636190000000,0.517648800000000;0.137339210000000,0.662251570000000,0.515571010000000;0.140209910000000,0.665859270000000,0.513426800000000;0.143302910000000,0.669458810000000,0.511215490000000;0.146616400000000,0.673049680000000,0.508936440000000;0.150147820000000,0.676631390000000,0.506588900000000;0.153894050000000,0.680203430000000,0.504172170000000;0.157851460000000,0.683765250000000,0.501685740000000;0.162015980000000,0.687316320000000,0.499129060000000;0.166383200000000,0.690856110000000,0.496501630000000;0.170948400000000,0.694384050000000,0.493802940000000;0.175706710000000,0.697899600000000,0.491032520000000;0.180653140000000,0.701402220000000,0.488189380000000;0.185782660000000,0.704891330000000,0.485273260000000;0.191090180000000,0.708366350000000,0.482283950000000;0.196570630000000,0.711826680000000,0.479221080000000;0.202219020000000,0.715271750000000,0.476084310000000;0.208030450000000,0.718700950000000,0.472873300000000;0.214000150000000,0.722113710000000,0.469587740000000;0.220123810000000,0.725509450000000,0.466226380000000;0.226396900000000,0.728887530000000,0.462789340000000;0.232814980000000,0.732247350000000,0.459276750000000;0.239373900000000,0.735588280000000,0.455688380000000;0.246069680000000,0.738909720000000,0.452024050000000;0.252898510000000,0.742211040000000,0.448283550000000;0.259856760000000,0.745491620000000,0.444466730000000;0.266941270000000,0.748750840000000,0.440572840000000;0.274149220000000,0.751988070000000,0.436600900000000;0.281476810000000,0.755202660000000,0.432552070000000;0.288921020000000,0.758393990000000,0.428426260000000;0.296478990000000,0.761561420000000,0.424223410000000;0.304147960000000,0.764704330000000,0.419943460000000;0.311925340000000,0.767822070000000,0.415586380000000;0.319808600000000,0.770914030000000,0.411152150000000;0.327795800000000,0.773979530000000,0.406640110000000;0.335885390000000,0.777017900000000,0.402049170000000;0.344074110000000,0.780028550000000,0.397381030000000;0.352359850000000,0.783010860000000,0.392635790000000;0.360740530000000,0.785964190000000,0.387813530000000;0.369214200000000,0.788887930000000,0.382914380000000;0.377778920000000,0.791781460000000,0.377938500000000;0.386432820000000,0.794644150000000,0.372886060000000;0.395174080000000,0.797475410000000,0.367757260000000;0.404001010000000,0.800274610000000,0.362552230000000;0.412913500000000,0.803040990000000,0.357268930000000;0.421908130000000,0.805774120000000,0.351910090000000;0.430983170000000,0.808473430000000,0.346476070000000;0.440136910000000,0.811138360000000,0.340967300000000;0.449367630000000,0.813768350000000,0.335384260000000;0.458673620000000,0.816362880000000,0.329727490000000;0.468053140000000,0.818921430000000,0.323997610000000;0.477504460000000,0.821443510000000,0.318195290000000;0.487025800000000,0.823928620000000,0.312321330000000;0.496615360000000,0.826376330000000,0.306376610000000;0.506271300000000,0.828786210000000,0.300362110000000;0.515991820000000,0.831157840000000,0.294278880000000;0.525776220000000,0.833490640000000,0.288126500000000;0.535621100000000,0.835784520000000,0.281908320000000;0.545524400000000,0.838039180000000,0.275626020000000;0.555483970000000,0.840254370000000,0.269281470000000;0.565497600000000,0.842429900000000,0.262876830000000;0.575562970000000,0.844565610000000,0.256414570000000;0.585677720000000,0.846661390000000,0.249897480000000;0.595839340000000,0.848717220000000,0.243328780000000;0.606045280000000,0.850733100000000,0.236712140000000;0.616292830000000,0.852709120000000,0.230051790000000;0.626579230000000,0.854645430000000,0.223352580000000;0.636901570000000,0.856542260000000,0.216620120000000;0.647256850000000,0.858399910000000,0.209860860000000;0.657641970000000,0.860218780000000,0.203082290000000;0.668053690000000,0.861999320000000,0.196293070000000;0.678488680000000,0.863742110000000,0.189503260000000;0.688943510000000,0.865447790000000,0.182724550000000;0.699414630000000,0.867117110000000,0.175970550000000;0.709898420000000,0.868750920000000,0.169257120000000;0.720391150000000,0.870350150000000,0.162602730000000;0.730889020000000,0.871915840000000,0.156028940000000;0.741388030000000,0.873449180000000,0.149561010000000;0.751884140000000,0.874951430000000,0.143228280000000;0.762373420000000,0.876423920000000,0.137064490000000;0.772851830000000,0.877868080000000,0.131108640000000;0.783315350000000,0.879285450000000,0.125405380000000;0.793759940000000,0.880677630000000,0.120005320000000;0.804181590000000,0.882046320000000,0.114965050000000;0.814576340000000,0.883393290000000,0.110346780000000;0.824940280000000,0.884720360000000,0.106217240000000;0.835269590000000,0.886029430000000,0.102645900000000;0.845560560000000,0.887322430000000,0.0997021900000000;0.855809600000000,0.888601340000000,0.0974518600000000;0.866013250000000,0.889868150000000,0.0959527700000000;0.876168240000000,0.891124870000000,0.0952504600000000;0.886271460000000,0.892373530000000,0.0953743900000000;0.896320020000000,0.893616140000000,0.0963353800000000;0.906311210000000,0.894854670000000,0.0981249600000000;0.916242120000000,0.896091270000000,0.100716800000000;0.926105790000000,0.897329770000000,0.104070670000000;0.935904440000000,0.898570400000000,0.108130940000000;0.945636260000000,0.899815000000000,0.112837730000000;0.955299720000000,0.901065340000000,0.118128320000000;0.964893530000000,0.902323110000000,0.123940510000000;0.974416650000000,0.903589910000000,0.130214940000000;0.983868290000000,0.904867260000000,0.136896710000000;0.993247890000000,0.906156570000000,0.143936200000000];
		
		% jet colormap that clips 0 values
		cmap =[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000;1,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';
		
		[bandwidth1,density1,X1,Y1] = kde2d_set_kernel(F, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y); % make bivariate KDE
		[bandwidth2,density2,X2,Y2] = kde2d_set_kernel(G, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y); % make bivariate KDE
		density1 = density1./sum(sum(density1)); %normalize so volume integrates to 1
		density2 = density2./sum(sum(density2)); %normalize so volume integrates to 1
		
		%make density contours
		%max_density1 = max(max(density1));
		%max_density_conf1 = max_density1 - max_density1*0.99;
		%max_density2 = max(max(density2));
		%max_density_conf2 = max_density2 - max_density2*0.99;
		
		%2D CDFs
		cdf2d1 = cumsum(cumsum(density1,1),2);    % take the CDF of x at y = max (y)
		CDF1_Q1 = cdf2d1;                                            %Local CDF for Quadrant 1
		CDF1_Q2 = cumsum(cumsum(density1, 1,'reverse'), 2);            %Local CDF for Quadrant 2
		CDF1_Q3 = cumsum(cumsum(density1, 1,'reverse'), 2, 'reverse'); %Local CDF for Quadrant 3
		CDF1_Q4 = cumsum(cumsum(density1, 1), 2, 'reverse');           %Loval CDF for Quadrant 4
		cdf2d2 = cumsum(cumsum(density2,1),2);    % take the CDF of x at y = max (y)
		CDF2_Q1 = cdf2d2;                                            %Local CDF for Quadrant 1
		CDF2_Q2 = cumsum(cumsum(density2, 1,'reverse'), 2);            %Local CDF for Quadrant 2
		CDF2_Q3 = cumsum(cumsum(density2, 1,'reverse'), 2, 'reverse'); %Local CDF for Quadrant 3
		CDF2_Q4 = cumsum(cumsum(density2, 1), 2, 'reverse');           %Loval CDF for Quadrant 4
		
		%make density contours
		max_density1 = max(max(density1)); 
		max_density_conf1 = max_density1 - max_density1*0.99;
		max_density2 = max(max(density2)); 
		max_density_conf2 = max_density2 - max_density2*0.99;
		
		%calculate 1D comparisons
		R1D(k,j) = r2(kde1,kde2); %Cross correlation (Saylor et al., 2012)
		L1D(k,j) = 1 - sum(abs(kde1-kde2)./2); %Likeness (Satkoski et al., 2013)
		S1D(k,j) = sum(((kde1.*kde2).^0.5)); %Similarity (Gehrels, 2000)
		[~,~,D1D(k,j)] = kstest2(F(:,1),G(:,1));
		[~,V1D(k,j)] = kuipertest2c(F(:,1),G(:,1));	
		
		%calculate 2D comparisons
		d1 = reshape(density1,size(density1,1)*size(density1,2),1);
		d2 = reshape(density2,size(density2,1)*size(density2,2),1);
		R2D(k,j) = ((sum((d1 - mean(d1)).*(d2 - mean(d2))))/(sqrt((sum((d1 - mean(d1)).*(d1 - mean(d1))))*(sum((d2 - mean(d2)).*(d2 - mean(d2)))))))^2;
		for m = 1:size(density1,1)
			for n = 1:size(density1,2)
				L2Dmap(m,n) = (abs(density1(m,n)-density2(m,n))/2); %Likeness map
				S2Dmap(m,n) = sqrt(density1(m,n).*density2(m,n)); % Similarity map
			end
		end
		L2D(k,j) = 1 - sum(sum(L2Dmap));
		S2D(k,j) = sum(sum(S2Dmap));
		D2Dtmp(1,1) = max(max(abs(CDF1_Q1 - CDF2_Q1),[],1)); %Maximum absolute difference for Quadrant 1
		D2Dtmp(1,2) = max(max(abs(CDF1_Q2 - CDF2_Q2),[],1)); %Maximum absolute difference for Quadrant 2
		D2Dtmp(1,3) = max(max(abs(CDF1_Q3 - CDF2_Q3),[],1)); %Maximum absolute difference for Quadrant 3
		D2Dtmp(1,4) = max(max(abs(CDF1_Q4 - CDF2_Q4),[],1)); %Maximum absolute difference for Quadrant 4
		D2Dtmp(1,5) = max(max(abs(CDF1_Q1 - CDF2_Q1),[],2)); %Maximum absolute difference for Quadrant 1
		D2Dtmp(1,6) = max(max(abs(CDF1_Q2 - CDF2_Q2),[],2)); %Maximum absolute difference for Quadrant 2
		D2Dtmp(1,7) = max(max(abs(CDF1_Q3 - CDF2_Q3),[],2)); %Maximum absolute difference for Quadrant 3
		D2Dtmp(1,8) = max(max(abs(CDF1_Q4 - CDF2_Q4),[],2)); %Maximum absolute difference for Quadrant 4
		D2D(k,j) = max(D2Dtmp);
		V2Dtmp(1,1) = max( max(CDF1_Q1 - CDF2_Q1,[],1) + max(CDF2_Q1 - CDF1_Q1,[],1) );
		V2Dtmp(1,2) = max( max(CDF1_Q2 - CDF2_Q2,[],1) + max(CDF2_Q2 - CDF1_Q2,[],1) );
		V2Dtmp(1,3) = max( max(CDF1_Q3 - CDF2_Q3,[],1) + max(CDF2_Q3 - CDF1_Q3,[],1) );
		V2Dtmp(1,4) = max( max(CDF1_Q4 - CDF2_Q4,[],1) + max(CDF2_Q4 - CDF1_Q4,[],1) );
		V2Dtmp(1,5) = max( max(CDF1_Q1 - CDF2_Q1,[],2) + max(CDF2_Q1 - CDF1_Q1,[],2) );
		V2Dtmp(1,6) = max( max(CDF1_Q2 - CDF2_Q2,[],2) + max(CDF2_Q2 - CDF1_Q2,[],2) );
		V2Dtmp(1,7) = max( max(CDF1_Q3 - CDF2_Q3,[],2) + max(CDF2_Q3 - CDF1_Q3,[],2) );
		V2Dtmp(1,8) = max( max(CDF1_Q4 - CDF2_Q4,[],2) + max(CDF2_Q4 - CDF1_Q4,[],2) );
		V2D(k,j) = max(V2Dtmp);
		
		
		k
		
		
	end
	
end


figure;
plot(x,kde1,'Color','b','LineWidth',3) %KDE scaled to the histogram
xlim([xmin xmax])

figure;
plot(x,kde2,'Color','b','LineWidth',3) %KDE scaled to the histogram
xlim([xmin xmax])

figure;
surf(X1,Y1,density1);
colormap(cmap)
shading interp
view(2)
grid off
%axis square
axis([xmin xmax -41 ymax])

figure;
surf(X1,Y1,density2);
colormap(cmap)
shading interp
view(2)
grid off
%axis square
axis([xmin xmax -41 ymax])

%{
figure
contour3(X1,Y1,density1,[max_density_conf1 max_density_conf1],'b', 'LineWidth', 12);
axis([xmin xmax ymin ymax])
view(2)

figure
contour3(X2,Y2,density2,[max_density_conf2 max_density_conf2],'b', 'LineWidth', 12);
axis([xmin xmax ymin ymax])
view(2)
%}



figure
hold on

plot(interval:interval:interval*max(j),median(R1D),'linewidth', 16,'color','g')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(R1D)+std(R1D)); (median(R1D)-std(R1D))], 'Color', 'g', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(R1D),5000,'s','filled','markeredgecolor','k','markerfacecolor','g')

plot(interval:interval:interval*max(j),median(L1D),'linewidth', 16,'color','m')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(L1D)+std(L1D)); (median(L1D)-std(L1D))], 'Color', 'm', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(L1D),5000,'s','filled','markeredgecolor','k','markerfacecolor','m')

plot(interval:interval:interval*max(j),median(S1D),'linewidth', 16,'color','c')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(S1D)+std(S1D)); (median(S1D)-std(S1D))], 'Color', 'c', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(S1D),5000,'s','filled','markeredgecolor','k','markerfacecolor','c')

plot(interval:interval:interval*max(j),median(D1D),'linewidth', 16,'color','b')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(D1D)+std(D1D)); (median(D1D)-std(D1D))], 'Color', 'b', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(D1D),5000,'s','filled','markeredgecolor','k','markerfacecolor','b')

plot(interval:interval:interval*max(j),median(V1D),'linewidth', 16,'color','r')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(V1D)+std(V1D)); (median(V1D)-std(V1D))], 'Color', 'r', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(V1D),5000,'s','filled','markeredgecolor','k','markerfacecolor','r')

axis([0 interval*max(j)+interval 0 1])
xlabel('n')
ylabel('Measure')
title('1D Measures')




figure
hold on

plot(interval:interval:interval*max(j),median(R2D),'linewidth', 16,'color','g')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(R2D)+std(R2D)); (median(R2D)-std(R2D))], 'Color', 'g', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(R2D),5000,'s','filled','markeredgecolor','k','markerfacecolor','g')

plot(interval:interval:interval*max(j),median(L2D),'linewidth', 16,'color','m')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(L2D)+std(L2D)); (median(L2D)-std(L2D))], 'Color', 'm', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(L2D),5000,'s','filled','markeredgecolor','k','markerfacecolor','m')

plot(interval:interval:interval*max(j),median(S2D),'linewidth', 16,'color','c')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(S2D)+std(S2D)); (median(S2D)-std(S2D))], 'Color', 'c', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(S2D),5000,'s','filled','markeredgecolor','k','markerfacecolor','c')

plot(interval:interval:interval*max(j),median(D2D),'linewidth', 16,'color','b')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(D2D)+std(D2D)); (median(D2D)-std(D2D))], 'Color', 'b', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(D2D),5000,'s','filled','markeredgecolor','k','markerfacecolor','b')

plot(interval:interval:interval*max(j),median(V2D),'linewidth', 16,'color','r')
plot([(interval:interval:interval*max(j)); (interval:interval:interval*max(j))], [(median(V2D)+std(V2D)); (median(V2D)-std(V2D))], 'Color', 'r', 'LineWidth',12) % Error bars
scatter(interval:interval:interval*max(j),median(V2D),5000,'s','filled','markeredgecolor','k','markerfacecolor','r')

axis([0 interval*max(j)+interval 0 1])
xlabel('n')
ylabel('Measure')
title('2D Measures')









%% KDE function with Gaussian kernel
function [a] = kde(m, s, xmin, xmax, xint)
x = xmin:xint:xmax;
n = length(m);
f = zeros(n,length(x));
for i = 1:n;
	f(i,:) = (1./ (s(i)*sqrt(2*pi)) .* exp (  (-((x-m(i)).^2)) ./ (2*((s(i)).^2))  ).*xint); % Gaussian
end
a = (sum(f))/n; %sum and normalize
end

%% Cross correlation
function [b] = r2(x, y)

x = x;
y = y;

lengthx = length(x);

xmean = mean(x);
ymean = mean(y);

xcov = zeros(1,length(x));
ycov = zeros(1,length(y));

for i = 1:lengthx;
	xcov(i) = x(i) - xmean;
end

for i = 1:lengthx;
	ycov(i) = y(i) - ymean;
end

xcov = transpose (xcov);
ycov = transpose (ycov);

mult = xcov.*ycov;
numerator = sum(mult);

xcov2 = xcov.*xcov;
sumxcov2 = sum(xcov2);
ycov2 = ycov.*ycov;
sumycov2 = sum(ycov2);
mult2 = sumxcov2*sumycov2;
denominator = sqrt(mult2);

r = numerator/denominator;

b = r*r;

end

%% 2D KDE
%algorithm and function follows, all sourced from
%https://github.com/rctorres/Matlab/blob/master/kde2d.m or
%https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/17204/versions/5/previews/kde2d.m/index.html
%Modified by Sundell -- added bandwidth_x and bandwidth_y to the original code for set bandwidths (i.e., no Botev et al. (2010) algorithm)
function [bandwidth,density,X,Y]=kde2d_set_kernel(data,n,MIN_XY,MAX_XY,bandwidth_x,bandwidth_y)
% bivariate kernel density estimator
% with diagonal bandwidth matrix.
% The kernel is assumed to be Gaussian.
% The two bandwidth parameters are
% chosen optimally without ever
% using/assuming a parametric model for the data or any "rules of thumb".
% Unlike many other procedures, this one
% is immune to accuracy failures in the estimation of
% multimodal densities with widely separated modes (see examples).
% INPUTS: data - an N by 2 array with continuous data
%            n - size of the n by n grid over which the density is computed
%                n has to be a power of 2, otherwise n=2^ceil(log2(n));
%                the default value is 2^8;
% MIN_XY,MAX_XY- limits of the bounding box over which the density is computed;
%                the format is:
%                MIN_XY=[lower_Xlim,lower_Ylim]
%                MAX_XY=[upper_Xlim,upper_Ylim].
%                The dafault limits are computed as:
%                MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
%                MAX_XY=MAX+Range/4; MIN_XY=MIN-Range/4;
% OUTPUT: bandwidth - a row vector with the two optimal
%                     bandwidths for a bivaroate Gaussian kernel;
%                     the format is:
%                     bandwidth=[bandwidth_X, bandwidth_Y];
%          density  - an n by n matrix containing the density values over the n by n grid;
%                     density is not computed unless the function is asked for such an output;
%              X,Y  - the meshgrid over which the variable "density" has been computed;
%                     the intended usage is as follows:
%                     surf(X,Y,density)
% Example (simple Gaussian mixture)
% clear all
%   % generate a Gaussian mixture with distant modes
%   data=[randn(500,2);
%       randn(500,1)+3.5, randn(500,1);];
%   % call the routine
%     [bandwidth,density,X,Y]=kde2d(data);
%   % plot the data and the density estimate
%     contour3(X,Y,density,50), hold on
%     plot(data(:,1),data(:,2),'r.','MarkerSize',5)
%
% Example (Gaussian mixture with distant modes):
%
% clear all
%  % generate a Gaussian mixture with distant modes
%  data=[randn(100,1), randn(100,1)/4;
%      randn(100,1)+18, randn(100,1);
%      randn(100,1)+15, randn(100,1)/2-18;];
%  % call the routine
%    [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,60])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
% Example (Sinusoidal density):
%
% clear all
%   X=rand(1000,1); Y=sin(X*10*pi)+randn(size(X))/3; data=[X,Y];
%  % apply routine
%  [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,70])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
%  Reference:
% Kernel density estimation via diffusion
% Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
% Annals of Statistics, Volume 38, Number 5, pages 2916-2957.

global N A2 I
if nargin<2
	n=2^8;
end
n=2^ceil(log2(n)); % round up n to the next power of 2;
N=size(data,1);
if nargin<3
	MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
	MAX_XY=MAX+Range/2; MIN_XY=MIN-Range/2;
end
scaling=MAX_XY-MIN_XY;
if N<=size(data,2)
	error('data has to be an N by 2 array where each row represents a two dimensional observation')
end
transformed_data=(data-repmat(MIN_XY,N,1))./repmat(scaling,N,1);
%bin the data uniformly using regular grid;
initial_data=ndhist(transformed_data,n);

% discrete cosine transform of initial data
a= dct2d(initial_data);

% now compute the optimal bandwidth^2
I=(0:n-1).^2; A2=a.^2;
t_star=root(@(t)(t-evolve(t)),N);
p_02=func([0,2],t_star);p_20=func([2,0],t_star); p_11=func([1,1],t_star);

t_x=(p_20^(3/4)/(4*pi*N*p_02^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);
t_y=(p_02^(3/4)/(4*pi*N*p_20^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);

%bandwidth_opt = sqrt([t_x,t_y]).*scaling;

% Sundell modified this bit for set kernels
bandwidth = [bandwidth_x, bandwidth_y];
t_x = (bandwidth_x(1,1)/scaling(1,1))^2;
t_y = (bandwidth_y(1,1)/scaling(1,2))^2;

% smooth the discrete cosine transform of initial data using t_star
a_t=exp(-(0:n-1)'.^2*pi^2*t_x/2)*exp(-(0:n-1).^2*pi^2*t_y/2).*a;

% now apply the inverse discrete cosine transform
if nargout>1
	density=idct2d(a_t)*(numel(a_t)/prod(scaling));
	density(density<0)=eps; % remove any negative density values
	[X,Y]=meshgrid(MIN_XY(1):scaling(1)/(n-1):MAX_XY(1),MIN_XY(2):scaling(2)/(n-1):MAX_XY(2));
end

end
%#######################################
function  [out,time]=evolve(t)
global N
Sum_func = func([0,2],t) + func([2,0],t) + 2*func([1,1],t);
time=(2*pi*N*Sum_func)^(-1/3);
out=(t-time)/time;
end
%#######################################
function out=func(s,t)
global N
if sum(s)<=4
	Sum_func=func([s(1)+1,s(2)],t)+func([s(1),s(2)+1],t); const=(1+1/2^(sum(s)+1))/3;
	time=(-2*const*K(s(1))*K(s(2))/N/Sum_func)^(1/(2+sum(s)));
	out=psi(s,time);
else
	out=psi(s,t);
end

end
%#######################################
function out=psi(s,Time)
global I A2
% s is a vector
w=exp(-I*pi^2*Time).*[1,.5*ones(1,length(I)-1)];
wx=w.*(I.^s(1));
wy=w.*(I.^s(2));
out=(-1)^sum(s)*(wy*A2*wx')*pi^(2*sum(s));
end
%#######################################
function out=K(s)
out=(-1)^s*prod((1:2:2*s-1))/sqrt(2*pi);
end
%#######################################
function data=dct2d(data)
% computes the 2 dimensional discrete cosine transform of data
% data is an nd cube
[nrows,ncols]= size(data);
if nrows~=ncols
	error('data is not a square array!')
end
% Compute weights to multiply DFT coefficients
w = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
weight=w(:,ones(1,ncols));
data=dct1d(dct1d(data)')';
	function transform1d=dct1d(x)
		
		% Re-order the elements of the columns of x
		x = [ x(1:2:end,:); x(end:-2:2,:) ];
		
		% Multiply FFT by weights:
		transform1d = real(weight.* fft(x));
	end
end
%#######################################
function data = idct2d(data)
% computes the 2 dimensional inverse discrete cosine transform
[nrows,ncols]=size(data);
% Compute wieghts
w = exp(i*(0:nrows-1)*pi/(2*nrows)).';
weights=w(:,ones(1,ncols));
data=idct1d(idct1d(data)');
	function out=idct1d(x)
		y = real(ifft(weights.*x));
		out = zeros(nrows,ncols);
		out(1:2:nrows,:) = y(1:nrows/2,:);
		out(2:2:nrows,:) = y(nrows:-1:nrows/2+1,:);
	end
end
%#######################################
function binned_data=ndhist(data,M)
% this function computes the histogram
% of an n-dimensional data set;
% 'data' is nrows by n columns
% M is the number of bins used in each dimension
% so that 'binned_data' is a hypercube with
% size length equal to M;
[nrows,ncols]=size(data);
bins=zeros(nrows,ncols);
for i=1:ncols
	[dum,bins(:,i)] = histc(data(:,i),[0:1/M:1],1);
	bins(:,i) = min(bins(:,i),M);
end
% Combine the  vectors of 1D bin counts into a grid of nD bin
% counts.
binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t=root(f,N)
% try to find smallest root whenever there is more than one
N=50*(N<=50)+1050*(N>=1050)+N*((N<1050)&(N>50));
tol=10^-12+0.01*(N-50)/1000;
flag=0;
while flag==0
	try
		t=fzero(f,[0,tol]);
		flag=1;
	catch
		tol=min(tol*2,.1); % double search interval
	end
	if tol==.1 % if all else fails
		t=fminbnd(@(x)abs(f(x)),0,.1); flag=1;
	end
end
end

%% cell2num
%sourced from https://www.mathworks.com/matlabcentral/fileexchange/15306-cell2num
function [outputmat]=cell2num(inputcell)
% Function to convert an all numeric cell array to a double precision array
% ********************************************
% Usage: outputmatrix=cell2num(inputcellarray)
% ********************************************
% Output matrix will have the same dimensions as the input cell array
% Non-numeric cell contest will become NaN outputs in outputmat
% This function only works for 1-2 dimensional cell arrays

if ~iscell(inputcell), error('Input cell array is not.'); end

outputmat=zeros(size(inputcell));

for c=1:size(inputcell,2)
	for r=1:size(inputcell,1)
		if isnumeric(inputcell{r,c})
			outputmat(r,c)=inputcell{r,c};
		else
			outputmat(r,c)=NaN;
		end
	end
end

end

function [A] = randsamp(data, n)
[y,x]=size(data);
k = randperm(y);
A = data(k(1:n),:);
end

function [p,V] = kuipertest2c(x1,x2)


x1 = sort(x1);
x2 = sort(x2);
n1 = length(x1(~isnan(x1)));
n2 = length(x2(~isnan(x2)));

binEdges    =  [-inf ; sort([x1;x2]) ; inf];

binCounts1  =  histc (x1 , binEdges, 1);
binCounts2  =  histc (x2 , binEdges, 1);

sumCounts1  =  cumsum(binCounts1)./sum(binCounts1);
sumCounts2  =  cumsum(binCounts2)./sum(binCounts2);

sampleCDF1  =  sumCounts1(1:end-1);
sampleCDF2  =  sumCounts2(1:end-1);

deltaCDF1  =  sampleCDF2 - sampleCDF1;
maxdeltaCDF1 = max(deltaCDF1);


deltaCDF2  =  sampleCDF1 - sampleCDF2;
maxdeltaCDF2 = max(deltaCDF2);

V = maxdeltaCDF1 + maxdeltaCDF2;

ne = ((n1*n2)/(n1+n2));
lambda  =  max((sqrt(ne) + 0.155 + (0.24/sqrt(ne))) * V);

if lambda<0.4  
p=1;  
h=0;
return
end

j=(1:100)';
pare=4*lambda*lambda*(j.^2)-1;
expo=exp(-2*lambda*lambda*(j.^2));
argo=pare.*expo;
p=2*sum(argo);

p = p;
V = V;

end
