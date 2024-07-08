function res = testLong_linearSysDT_observe_03_tank()
% testLong_linearSysDT_observe_03_tank - unit_test_function for guaranteed
% state estimation of linear discrete-time systems.
%
% Checks the solution of the linearSysDT class for a tank example;
% It is checked whether the enclosing interval of the final observed set 
% is close to an interval provided by a previous solution that has been saved
%
% Syntax:
%    res = testLong_linearSysDT_observe_03_tank
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false 

% Authors:       Matthias Althoff
% Written:       26-March-2021
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------


%% Load pedestrian model
load tankModel_lin_dim30 tank params options

% more robust case
params.tFinal = 10;
params.u = params.u(:,1:20);
params.y = params.y(:,1:20);

% Set of evaluated estimators
Estimator = {
    'FRad-A' 
    'FRad-B'
    'FRad-C'
    'ESO-A'
    'ESO-B'
    };

% init
resPartial = [];

% set accuracy
accuracy = 1e-6;

%% perform evaluation
% loop over estimators
for iEst = 1:length(Estimator)

    % set algorithm
    estName = Estimator{iEst};
    options.alg = estName;
    
    % if constrained zonotopes or ellipsoids are used
    paramsNew = params;
    if any(strcmp(estName,{'ESO-A','ESO-B','ESO-C','ESO-D','ESO-E'}))
        paramsNew.R0 = ellipsoid(params.R0);
        paramsNew.W = ellipsoid(params.W);
        paramsNew.V = ellipsoid(params.V);
    elseif any(strcmp(estName,{'CZN-A','CZN-B'}))
        paramsNew.R0 = conZonotope(params.R0);
    end
    
    % evaluate observer
    estSet = observe(tank,paramsNew,options);
    
    % enclose last estimated set by interval
    IH = interval(estSet.timePoint.set{end});
    
    % obtain enclosing intervals
    if strcmp(options.alg,'FRad-A')
        if params.tFinal == 10
            IH_saved = interval( ...
                [0.9752560707290552; 0.8115148164127659; -4.9119918443438184; 1.8406118995337666; -3.6413041903752053; 2.9211627298680911; 3.5621815238011942; -2.4393054485111523; -8.9931676949659956; -1.4687408162436451; -1.7097795786467691; 2.1548085204489360; 2.4256375089568718; 0.5645515479873555; -3.4660049169359635; 0.3093169940971032; 3.3358468192914250; 3.1676150200160729; 3.4401371812574664; 2.1420523188230196; 0.9194281150760770; -0.6561515163929893; 1.9745350289944139; -1.2713452629662374; -0.8629288589561362; 1.2178808420446461; -0.3024586144971499; 0.0530537881661066; 2.8107536616539512; -3.5100252216208183], ...
                [1.4384408009013263; 1.2910523869455435; 8.6891081501634790; 2.4542031950294794; -3.2133815641203975; 3.4077826537628564; 4.0576544336415754; -2.0186355681865416; 4.6083585057176588; -0.8551442855273086; -1.2847407574612177; 2.6913306167750428; 2.9128021657688232; 0.9852708318597780; 10.1351856853734859; 0.9229067443307553; 3.7636825901423849; 3.6822402141855095; 3.9290389425299046; 2.5724177986569186; 1.3437002346505698; -0.2281277425649266; 2.3992211270879999; -0.7527044426188683; -0.3716104197659312; 1.6463359492524483; 0.1256524719496209; 0.4810821363323661; 3.2309907999509617; 3.9695351203239344]);
        else
            IH_saved = interval( ...
                [24.8508520864847533; 26.1082577476961823; 9.7969983169726493; 4.4843927427937809; 1.1874849532827993; 0.6607731275444062; 1.3648724759299218; 0.4355593642171014; -1.8658152776442889; -0.6604706011308403; -1.4837210851951246; -0.2792919073416293; 0.3576212613524974; 1.1504381704636328; 1.3698023860886428; 2.5259656725247011; 2.8177677588913492; 2.3027216657361889; 2.8470270356020988; 3.6319457243077631; 2.3880493878791680; 1.8815363843081574; 1.1737615402686230; -0.0140909704744678; 0.5926713449496390; 0.9903287419735227; 1.1728197119395465; 1.5664827125769585; 1.6855743820810649; -0.4495824563850706], ...
                [25.7376390893745004; 27.0161996614473559; 12.7199912073956884; 5.5758300974532160; 2.0769317970053227; 1.6653719408526408; 2.3914410442634697; 1.3487503489268915; 1.1115335947294791; 0.4938444283956707; -0.5193234513181568; 0.8604636542584461; 1.3918804733840990; 2.0917834660576098; 4.2955828674724286; 3.6521123602094017; 3.7138542459786685; 3.3577682279672274; 3.8435418385752094; 4.3234995765789144; 3.1593076976193886; 2.7207312801720249; 2.0943161342465140; 1.0304110613337611; 1.5883347624129043; 1.7304066592061818; 1.8896373254490491; 2.2976896676078664; 2.3281969770720239; 3.2575260096359511]);
        end
    elseif strcmp(options.alg,'FRad-B')
        if params.tFinal == 10
            IH_saved = interval( ...
                [0.9680774036227109; 0.8071482135158163; -4.8222243840459988; 1.8507412771954819; -3.6407616352779746; 2.8990065068656330; 3.5434674013757670; -2.4392507963305516; -9.0133409039145231; -1.4703843753998438; -1.7100150681729285; 2.1382556790156739; 2.4122242989681939; 0.5646782532886737; -3.4065720205978569; 0.3191900431401611; 3.3366316770016571; 3.1517373994961910; 3.4277718008167892; 2.1422891039513909; 0.9186711399149523; -0.6567085448257329; 1.9749480694164343; -1.2918595385849798; -0.8807168266299665; 1.2173093790028053; -0.3032999396287579; 0.0527178406896560; 2.8106363192412012; -3.5136900929445312], ...
                [1.4589570964144061; 1.3091046393504677; 8.6673124041397518; 2.4551345616703788; -3.2133009666696841; 3.4126096221507316; 4.0604160622131893; -2.0182573188026969; 4.4736150156366481; -0.8660805925940589; -1.2853892869613319; 2.6987867577438545; 2.9183378243391078; 0.9855739078830034; 10.0800946950977455; 0.9234917050546980; 3.7640312415973227; 3.6910020833436223; 3.9363597704146787; 2.5736440100924205; 1.3440445449008971; -0.2276162990738820; 2.3995810919915521; -0.7489389912061850; -0.3700305446153762; 1.6470084486051584; 0.1259280008775757; 0.4818086796067666; 3.2309280686125463; 3.9972793658915511]);
        else
            IH_saved = interval( ...
                [24.8744894939771157; 26.1304072863435799; 9.8975136913275268; 4.5049231336015403; 1.1940964247543748; 0.6603229118796221; 1.3648122008182102; 0.4317464961533661; -1.8284478839928671; -0.6479858247409572; -1.4686094315165552; -0.2716896878531231; 0.3662336863589706; 1.1346762213886370; 1.3552306170299584; 2.5335859583113836; 2.8308538658861204; 2.3279409279444843; 2.8679132398732627; 3.6628748965542091; 2.4391465341948368; 1.9113379646540554; 1.1784772550823108; 0.0012172933303474; 0.6078120963582760; 1.0237219818586378; 1.1983385662287871; 1.5728751463140394; 1.6877932654216727; -0.4519737509515998], ...
                [25.7141852554934474; 26.9943005008587598; 12.6214877342402740; 5.5544230782608377; 2.0716091996564692; 1.6662381357031777; 2.3918505958651610; 1.3500722035364288; 1.0810096564401190; 0.4833129881164474; -0.5382746271196694; 0.8536550325349967; 1.3840933865450498; 2.1046354667397900; 4.3089664940964489; 3.6442301148984191; 3.7001091951524661; 3.3317733424931189; 3.8219186986291551; 4.2949932239265474; 3.1112676139806368; 2.6794839504592347; 2.0902487190766825; 1.0149430334326723; 1.5730593560711159; 1.6917026564258897; 1.8919424353583985; 2.2783937142942312; 2.3372761816881957; 3.2706449612946011]);
        end
    elseif strcmp(options.alg,'FRad-C')
        if params.tFinal == 10
            IH_saved = interval( ...
                [0.9573517114404899; 0.7922870888477151; -5.3037185456777216; 1.8109016927014889; -3.6449442617063985; 2.9190561849388215; 3.5596481718502453; -2.4429200292586306; -8.7036262757223870; -1.4601227009446205; -1.7133188312893226; 2.1653160061689571; 2.4351246254593581; 0.5641559559209275; -3.1810132934575801; 0.3177274041945076; 3.3338514802318961; 3.1694544851145210; 3.4416120680134075; 2.1354735840257164; 0.9263282629135323; -0.6532145927919958; 1.9842002793740323; -1.2732938768245488; -0.8652961650626394; 1.2180668922521118; -0.2969267109751242; 0.0564650528739660; 2.8189154298548833; -3.5093485295076370], ...
                [1.4258459319458539; 1.2781632049532512; 8.6762184318119147; 2.4610140486133578; -3.2159115121652437; 3.4110557225745972; 4.0614806815932463; -2.0210993732421012; 5.2766888036558441; -0.8100020356408006; -1.2874289426086605; 2.7069013344015720; 2.9276147095526284; 0.9860158477238019; 10.7989958303716733; 0.9678377054465791; 3.7628183379026972; 3.6891468280526318; 3.9360884696511569; 2.5668824758091078; 1.3510171225135450; -0.2245912604259325; 2.4095427923227635; -0.7493923027041971; -0.3682154827104058; 1.6472650658795212; 0.1317902714868420; 0.4851067933309699; 3.2401695992814150; 3.9701553984611202]);
        else
            IH_saved = interval( ...
                [24.8489884426172729; 26.1074432240818055; 9.6974448943767975; 4.4426736832374347; 1.1924859843998488; 0.5969863555739598; 1.2987541205551594; 0.4182724097728672; -1.9981633623521291; -0.6792549455873498; -1.4474245504909191; -0.3165645213044527; 0.3268181852685199; 1.1545976158863809; 1.2494139527807153; 2.5229343765534269; 2.8444922322184474; 2.2745382609855493; 2.8150275543772327; 3.6587385804359602; 2.4093981460122884; 1.8893163427180095; 1.1788106284277835; -0.0732506028855608; 0.5382474179834937; 1.0295731462663515; 1.1800862827389353; 1.5689949395245648; 1.6746261533997244; -0.4473175396889610], ...
                [25.7403546234748042; 27.0179056188779683; 12.8143493439754792; 5.6140739146597118; 2.0714327864266386; 1.7296478497796928; 2.4581235447281640; 1.3569515183003991; 1.2331613926884164; 0.5075945105477289; -0.5523082735708618; 0.8930354992870442; 1.4182025434820491; 2.0895499128761936; 4.4125241277713663; 3.6516171291620001; 3.6806705461137814; 3.3888060849798940; 3.8784208702225422; 4.2850280815455282; 3.1482399539386354; 2.6866623288279516; 2.0941323702225172; 1.0960866407062295; 1.6488501364097501; 1.6977737922610507; 1.8869882104392939; 2.2723431782621413; 2.3549226160128121; 3.2573418473607001]);
        end
    elseif strcmp(options.alg,'ESO-A')
        if params.tFinal == 10
            IH_saved = interval( ...
                [-15.2433976752504616; -16.3289779510016437; -21.4116073497350499; -14.4045068196781152; -17.2996428027256925; -14.9768715909510792; -15.2234305826016012; -16.5034830777695092; -21.5370184531002558; -15.9634523196519922; -16.1580494666462791; -16.0035396865744985; -14.4588410701886119; -15.0100138274007744; -21.3885427155441548; -15.1261747599701426; -13.8277430798768908; -15.4932268408418405; -14.6662497724544156; -14.4243364421870179; -14.7959633112674549; -15.5717633274791964; -14.3371745535941777; -17.2122869943483146; -16.3294939981438745; -14.6900163890945432; -15.4121551023438954; -15.2218188586844292; -13.9078589844149736; -21.3853096247184915], ...
                [17.3803974785988551; 18.1348633556177141; 21.5512926792568322; 16.4742619432173072; 13.9826070443154684; 17.7618386393762790; 19.0640793098383980; 14.3297780244884478; 21.4374227424770289; 14.9153901108131741; 14.6769651264428980; 18.0391319884982551; 17.1508607074085759; 15.8183559718676303; 21.5858973066112725; 15.7526180853497220; 17.4546912945500239; 18.0537994256025769; 17.9263269266741716; 16.8459798260367535; 16.0122235750704718; 15.2585474224535425; 16.4976222569547097; 16.3588626926099572; 16.2765269029197839; 16.1307484560549632; 15.4181566574762208; 15.6085027200512769; 16.9319059750737750; 21.6148507864855510]);
        else
            IH_saved = interval( ...
                [21.2133329217047653; 22.2350863057436001; 1.7108765685474943; -2.0575805135966672; -3.7606722777891761; -3.3932969741892469; -2.8879400152429375; -3.9514439617891917; -8.8613632217416072; -6.3403457183703802; -5.6710426266856224; -4.7706408322633793; -3.8089616914375855; -3.2891357418208136; -7.1139955914563480; -4.0324308000532110; -2.4815862233751429; -2.0541070360733449; -1.3861279369876360; -1.5714071769040117; -1.8474662444567982; -2.4132326254940355; -3.2207774130314073; -4.3556469430372955; -3.6136607015146289; -3.4385535540643630; -3.2332563629895601; -2.7921460646984295; -2.9838009061179420; -9.8908148868547983], ...
                [29.3822599808922895; 30.8969118261702249; 20.0853050926312910; 11.3529525052896183; 7.2000291025448906; 5.4964521232905446; 6.4064773880861843; 5.7536522842198465; 9.5207913833354869; 7.0763631206861293; 4.1445764442222899; 5.2601075806204927; 5.4664230169225076; 6.3867151051656927; 11.2661188246853179; 9.3776210852357167; 8.4962125077722987; 7.2575880267872837; 7.6362690581907433; 9.1056770414625241; 7.1932142251975755; 7.0387635806584736; 6.2828389759134700; 5.4005739714913412; 5.8256243893866362; 6.0559752698356597; 6.2477444754414959; 6.6923735369888311; 6.7221301909410318; 12.3896495966221885]);
        end
    elseif strcmp(options.alg,'ESO-B')
        if params.tFinal == 10
            IH_saved = interval( ...
                [-5.2038518167427252; -5.6624561571932155; -19.8612402701268032; -1.7633020228035958; -6.9203048201474715; -3.5201765359338872; -3.1376505018858478; -5.7359251947910952; -20.1125259945363979; -4.7215736935469774; -5.0211019518647975; -4.5030394561726901; -3.7298624040643769; -2.7654734951606654; -19.7706070038089052; -3.3568999436384539; -0.0803970239851748; -3.6308288483196387; -3.1280141611950367; -1.2825982112203365; -2.4534581359892220; -3.9271436102227701; -1.3207964481803782; -7.5159031189062828; -6.8948351583023104; -2.0899336537153044; -3.6030469607229181; -3.2347999663865235; -0.5597932114432229; -19.8267101372603172], ...
                [7.5581215727921265; 7.7022972158120879; 20.0339351222820561; 5.7094686662276182; 0.2269645264331905; 9.2856383310830797; 10.1647774499807948; 1.3314007939067607; 19.7788875931422012; 2.7511951493123830; 2.0264375010932358; 8.9673160690394607; 8.7130776532235750; 4.2996218275211877; 20.1208060419378185; 4.1158617042619063; 7.0668823419812306; 9.5857573726558627; 9.6309972590863353; 5.8823804899508048; 4.5879640612153905; 3.1194718261171737; 5.7263972065301907; 5.7029312103848726; 5.8653367494688879; 4.9751491877765721; 3.4435800014647020; 3.8118194244196344; 6.4876832074692210; 20.2771398023552898]);
        else
            IH_saved = interval( ...
                [20.7953998270651574; 21.8329455057183139; 1.3392278147180754; -3.1132476683980590; -4.1843843036073789; -3.6433072377125653; -3.1162868342062442; -4.0124551435412004; -9.4127369442282252; -7.5902217790161854; -6.1360392757940776; -4.7648045210306922; -3.7828536733778200; -3.2245661285995642; -7.1556542209921652; -5.1313335019746482; -2.6523461806590869; -2.1089545537173628; -1.4155211946456294; -1.3764993182473710; -1.8012160439254377; -2.3867577657859527; -3.0582223553287129; -4.4359913565194082; -3.6736100005025949; -3.4676417915685489; -3.1826321569559921; -2.7811210821045713; -2.7352961465581287; -11.0858297385081475], ...
                [29.7814579721877486; 31.2791108054232438; 20.4738801201625478; 12.6259601039052427; 7.2955722847581832; 5.8678239379119530; 6.7673502698164452; 5.7480596992606907; 9.5906350256138637; 8.1335526770628981; 4.4086326452722400; 5.3641926718408524; 5.5455778626798526; 6.4490095974704573; 11.8345793777025321; 10.5783085370278904; 8.8288408037307136; 7.5939322104039633; 7.9386643288884642; 9.2157698301927695; 7.3278164513857913; 6.9887768734837223; 6.3549667973044333; 5.5033872218367863; 5.9038966381807878; 6.1972246462544289; 6.2645325946859218; 6.6267599758555207; 6.7139210585796079; 13.8471421536522268]);
        end
    end

    %check if slightly bloated versions enclose each other
    resPartial(end+1) = isequal(IH,IH_saved,accuracy);
end

% final result
res = all(resPartial);

% ------------------------------ END OF CODE ------------------------------
