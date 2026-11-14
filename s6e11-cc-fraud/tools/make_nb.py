"""Build the first public notebook for Playground S6E11 (credit card fraud, ROC AUC).

    python tools/make_nb.py            # writes kernels/s6e11-first-look/s6e11-first-look.ipynb
The notebook runs on Kaggle (/kaggle/input/...) and locally (data/), CPU only.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/s6e11-first-look"
TITLE = "Credit Card Fraud | LB 0.9537 | Shallow-Tree Stack"
SLUG = "credit-card-fraud-lb-0-9537-shallow-tree-stack"
VERSION = "v29"                                 # bump with every push: members trained in this version are recorded under it
assert len(TITLE) <= 50
# v29: the weights of four public mixes, copied from areebbb's 'Bold, Then Bolder' (bolder, LB 0.95388) and goodpjw2008's 'LM Ladder | Public and CV Up' (bold 0.95386, balanced 0.95380, careful 0.95376)
R29_JSON = '{"bolder": {"s:alex_glm_gen": 0.004659655001642133, "b:lgb_diglead_l3": 0.07037499081158276, "s:daoviet_gpdvae": 0.16759272015867438, "b:realmlp_v1": 0.028265988392542088, "b:gb_bag10": -0.018152655535374172, "b:lgb_dig_l7": 0.026162003105571548, "b:lgb_dig_s100_bag5": 0.023205824985217123, "s:lm7_5fold": -0.04643526723210693, "b:lgb_te_l4_s300": 0.012845944273216886, "b:lgb_diglead_bag": -0.008520524531314792, "s:masaya_gpt2": -0.06938586034217065, "b:lgb_dig_s100": -0.06905761136150596, "b:gbx_bag3": -0.025385082895315353, "b:lgb_te_l4": 0.035196443631714015, "s:sadam_v6_gb_xgb": -0.13553456267863312, "b:lgb_dig_l4": 0.04984890049165017, "b:glm_sb": -0.015572335325911311, "b:glm_dl_s50_bag": -0.043295376575860016, "b:lgb_te_l4_s100": 0.010731651329495398, "b:glm_ohk_c01": -0.0005446893557200782, "b:gbn_bag5_w512": 0.0551862603361086, "b:glm_ohk": -0.022950181793756356, "b:glm_dl_bag": 0.015527551247048318, "b:realmlp_dig_s100": 0.019529287294532443, "b:gbr_bag3": -0.007872454837865235, "wide": -0.0567073526172608, "s:gbn_bag5_w512": 0.024585711783519787, "b:xgb_dig_d2": -0.04001405009889738, "b:ebm_i25_bag4": 0.052776831945719284, "s:megayak_shallow_tree": 0.3676405944329277, "s:gb_bag5": 0.0022942519878240766, "b:lgb_dig_s100_bag": -0.0031388296177910285, "s:gpt2_seed2026": -0.02172163397492512, "b:gbe_bag3": 0.016771631110015763, "s:yekenot_realmlp": -0.018220789336614255, "s:sadam_tabpfn_100k_tp3": -0.08146952344397645, "s:sadam_v6_gb_lgb": 0.09202978068795398, "b:glm_ohk2": -0.010746902043578546, "b:gb_ohk": -0.0403299060795074, "b:ebm_dig": -0.025668026406425494, "b:ebm_diglead_r11": 0.021711464683545754, "b:xgb_te_d2": -0.017507176405797825, "b:realmlp_rawdig": -0.03995919213462431, "b:gbn_bag5_w128": -0.03944938723524659, "s:sadam_v4_cat": 0.044170457123327754, "b:gb_l3": -0.08781795649166141, "b:lgb_diglead_bag5": -0.0691076844266097, "b:ebm_diglead_r13": 0.03797203360211246, "b:glm_dl": -0.0066911606091787906, "b:ebm_all_bag": -0.05809235897307825, "b:ebm_diglead_bag": -0.011299892151034639, "b:lgb_dig_s100_l3_bag": -0.017863982273807155, "b:realmlp_te_s2": -0.0007511365624976998, "b:realmlp_cnt": 0.010152092525701466, "b:ebm_i25_bag": 0.022755487867529373, "b:lgb_dig_s100_l3": 0.054036258199026535, "b:glm_dig": 0.004367002340381022, "b:lgb_diglead_l2": -0.060799370916732313, "b:lgb_diglead_l3_bag": -0.016998031665317204, "b:lgb_rawdig_l4": 0.018367968146419205, "b:realmlp_te_cnt": -0.0029453567331327295, "b:ebm_dig_r7": 0.04840096288719489, "b:gb_bag5": 0.033276434668693684, "b:gbx_bag5": 0.007256671781074796, "b:glm_ohkl": 0.010941852623233855, "s:tabpfn_slimypunk_tp1": -0.05059762171891728, "b:glm_dl_s50": -0.06411999038593352, "s:sameer_glm10f": 0.09096559940413664, "b:ebm_diglead_i25": -0.020509275358757188, "b:lgb_te_l4_s5": 0.011694566527880524, "b:ebm_dig_bag": -0.02860322960251588, "b:lgb_raw_l31": -0.00015178805153019635, "b:glm_ohk2_bag": -0.03766304548255879, "b:lgb_raw_l4": 0.0091529109828078, "b:xgb_raw_d3": 0.018532570780381338, "b:ebm_lead4_bag": -0.01900597337730603, "s:megayak_5f_lm": -0.1500914472131036, "b:glm_te": -0.005973898251156665, "b:glm_ohkl_bag": 0.021686855681833317, "b:realmlp_allcat": -0.014402601850564621, "b:lgb_raw_l4_cnt": -0.08232013696341364, "b:realmlp_dig": -0.034559470951024046, "b:lgb_dk_l4": 0.013571307927505006, "b:realmlp_dig_s2": 0.018038024291734817, "b:lgb_diglead_l4": 0.006437648823683151, "b:realmlp_te": 0.016229843827016463, "b:gbc_bag3": -0.016244727973699344}, "bold": {"s:megayak_5f_lm": -0.12205919443860914, "s:masaya_gpt2": -0.06997547490341371, "s:lm7_5fold": -0.040956430057266076, "s:gpt2_seed2026": -0.012508511564926699, "s:tabpfn_slimypunk_tp1": -0.039316699160350764, "s:sadam_tabpfn_100k_tp3": -0.061677762652753675, "s:yekenot_realmlp": -0.014637185195237004, "s:alex_glm_gen": 0.004537993603062768, "s:daoviet_gpdvae": 0.1359104270990657, "s:gbn_bag5_w512": 0.015432253915041055, "s:sameer_glm10f": 0.06272836350651828, "s:sadam_v6_gb_xgb": -0.10228295915095288, "s:sadam_v6_gb_lgb": 0.06627713517089104, "s:sadam_v4_cat": 0.03126101632182668, "s:gb_bag5": 0.008855983394809901, "s:megayak_shallow_tree": 0.28129985621486264, "wide": -0.034852407295305754, "b:ebm_all_bag": -0.050611080400723436, "b:ebm_dig": -0.016392951045769617, "b:ebm_dig_bag": -0.021971929033966105, "b:ebm_dig_r7": 0.03715898221284049, "b:ebm_diglead_bag": -0.011280640758210015, "b:ebm_diglead_i25": -0.01760600397929515, "b:ebm_diglead_r11": 0.020606132176681696, "b:ebm_diglead_r13": 0.031952532024431754, "b:ebm_i25_bag": 0.019937230510505454, "b:ebm_i25_bag4": 0.04011014273129195, "b:ebm_lead4_bag": -0.016285931935351022, "b:gb_bag10": -0.017390237028207023, "b:gb_bag5": 0.026782938331671603, "b:gb_l3": -0.06609634322241796, "b:gb_ohk": -0.02882833922678926, "b:gbc_bag3": -0.012871505523924395, "b:gbe_bag3": 0.012674094986094425, "b:gbn_bag5_w128": -0.029120675300013294, "b:gbn_bag5_w512": 0.04504705483635584, "b:gbr_bag3": -0.005251606394839109, "b:gbx_bag3": -0.01901561844474828, "b:gbx_bag5": 0.006799891770028022, "b:glm_dig": 0.003088377583964293, "b:glm_dl": -0.006428517981988452, "b:glm_dl_bag": 0.01502056793927036, "b:glm_dl_s50": -0.0501984153162254, "b:glm_dl_s50_bag": -0.028630562092033966, "b:glm_ohk": -0.017070904616440657, "b:glm_ohk2": -0.004903067440865727, "b:glm_ohk2_bag": -0.029877217995289047, "b:glm_ohk_c01": 0.0003448885987200072, "b:glm_ohkl": 0.006673473442067076, "b:glm_ohkl_bag": 0.01766365615186144, "b:glm_sb": -0.01145379830324528, "b:glm_te": -0.004792258197303907, "b:lgb_dig_l4": 0.0366960138715965, "b:lgb_dig_l7": 0.01949687340453925, "b:lgb_dig_s100": -0.051481356730372616, "b:lgb_dig_s100_bag": -7.44946546564366e-05, "b:lgb_dig_s100_bag5": 0.016566984017469158, "b:lgb_dig_s100_l3": 0.039879653631481535, "b:lgb_dig_s100_l3_bag": -0.014332571073509623, "b:lgb_diglead_bag": -0.00442949281469811, "b:lgb_diglead_bag5": -0.05069793900237127, "b:lgb_diglead_l2": -0.0434403013635199, "b:lgb_diglead_l3": 0.051384825410253424, "b:lgb_diglead_l3_bag": -0.014548087315257047, "b:lgb_diglead_l4": 0.005059024340666011, "b:lgb_dk_l4": 0.010395272568768744, "b:lgb_raw_l31": 6.272373140902098e-05, "b:lgb_raw_l4": 0.006062225361634528, "b:lgb_raw_l4_cnt": -0.060074422527671846, "b:lgb_rawdig_l4": 0.012821826099948688, "b:lgb_te_l4": 0.026453964192257394, "b:lgb_te_l4_s100": 0.007166689614341422, "b:lgb_te_l4_s300": 0.009325395175060891, "b:lgb_te_l4_s5": 0.00877347694863135, "b:realmlp_allcat": -0.010498730399732318, "b:realmlp_cnt": 0.00748969325179405, "b:realmlp_dig": -0.026045115561739243, "b:realmlp_dig_s100": 0.01482032375010226, "b:realmlp_dig_s2": 0.012780340101831603, "b:realmlp_rawdig": -0.029224792823957325, "b:realmlp_te": 0.011902655678168103, "b:realmlp_te_cnt": -0.0016811066269879565, "b:realmlp_te_s2": -0.00033527900132535184, "b:realmlp_v1": 0.02099235698670545, "b:xgb_dig_d2": -0.027214651902440812, "b:xgb_raw_d3": 0.013799277379315805, "b:xgb_te_d2": -0.012792356993134408}, "balanced": {"s:megayak_5f_lm": -0.0659946888896202, "s:masaya_gpt2": -0.07115470402589986, "s:lm7_5fold": -0.029998755707584368, "s:gpt2_seed2026": 0.005917733255070143, "s:tabpfn_slimypunk_tp1": -0.016754854043217735, "s:sadam_tabpfn_100k_tp3": -0.02209424107030813, "s:yekenot_realmlp": -0.007469976912482503, "s:alex_glm_gen": 0.004294670805904037, "s:daoviet_gpdvae": 0.07254584097984837, "s:gbn_bag5_w512": -0.002874661821916413, "s:sameer_glm10f": 0.006253891711281581, "s:sadam_v6_gb_xgb": -0.035779752095592386, "s:sadam_v6_gb_lgb": 0.014771844136765163, "s:sadam_v4_cat": 0.005442134718824526, "s:gb_bag5": 0.02197944620878155, "s:megayak_shallow_tree": 0.10861837977873254, "wide": 0.008857483348604336, "b:ebm_all_bag": -0.035648523256013816, "b:ebm_dig": 0.0021571996755421373, "b:ebm_dig_bag": -0.00870932789686656, "b:ebm_dig_r7": 0.014675020864131676, "b:ebm_diglead_bag": -0.011242137972560769, "b:ebm_diglead_i25": -0.011799461220371074, "b:ebm_diglead_r11": 0.018395467162953583, "b:ebm_diglead_r13": 0.019913528869070343, "b:ebm_i25_bag": 0.014300715796457614, "b:ebm_i25_bag4": 0.014776764302437287, "b:ebm_lead4_bag": -0.010845849051441005, "b:gb_bag10": -0.01586540001387272, "b:gb_bag5": 0.013795945657627443, "b:gb_l3": -0.022653116683931083, "b:gb_ohk": -0.005825205521352985, "b:gbc_bag3": -0.006125060624374499, "b:gbe_bag3": 0.004479022738251753, "b:gbn_bag5_w128": -0.008463251429546712, "b:gbn_bag5_w512": 0.02476864383685032, "b:gbr_bag3": -9.909508786856773e-06, "b:gbx_bag3": -0.006276689543614139, "b:gbx_bag5": 0.005886331747934474, "b:glm_dig": 0.0005311280711308347, "b:glm_dl": -0.0059032327276077766, "b:glm_dl_bag": 0.014006601323714443, "b:glm_dl_s50": -0.022355265176809174, "b:glm_dl_s50_bag": 0.0006990668756181338, "b:glm_ohk": -0.005312350261809259, "b:glm_ohk2": 0.00678460176455991, "b:glm_ohk2_bag": -0.014305563020749566, "b:glm_ohk_c01": 0.002124044507600178, "b:glm_ohkl": -0.0018632849202664824, "b:glm_ohkl_bag": 0.009617257091917682, "b:glm_sb": -0.0032167242579132175, "b:glm_te": -0.0024289780895983912, "b:lgb_dig_l4": 0.010390240631489167, "b:lgb_dig_l7": 0.006166614002474661, "b:lgb_dig_s100": -0.016328847468105947, "b:lgb_dig_s100_bag": 0.006054175271612747, "b:lgb_dig_s100_bag5": 0.0032893020819732264, "b:lgb_dig_s100_l3": 0.01156644449639154, "b:lgb_dig_s100_l3_bag": -0.0072697486729145595, "b:lgb_diglead_bag": 0.003752570618535255, "b:lgb_diglead_bag5": -0.01387844815389441, "b:lgb_diglead_l2": -0.008722162257095078, "b:lgb_diglead_l3": 0.01340449460759475, "b:lgb_diglead_l3_bag": -0.009648198615136732, "b:lgb_diglead_l4": 0.0023017753746317295, "b:lgb_dk_l4": 0.00404320185129622, "b:lgb_raw_l31": 0.0004917472972874557, "b:lgb_raw_l4": -0.00011914588071201624, "b:lgb_raw_l4_cnt": -0.015582993656188236, "b:lgb_rawdig_l4": 0.001729542007007652, "b:lgb_te_l4": 0.008969005313344148, "b:lgb_te_l4_s100": 3.676618403346857e-05, "b:lgb_te_l4_s300": 0.0022842969787489017, "b:lgb_te_l4_s5": 0.002931297790132998, "b:realmlp_allcat": -0.0026909874980677114, "b:realmlp_cnt": 0.0021648947039792175, "b:realmlp_dig": -0.009016404783169642, "b:realmlp_dig_s100": 0.005402396661241892, "b:realmlp_dig_s2": 0.0022649717220251738, "b:realmlp_rawdig": -0.007755994202623359, "b:realmlp_te": 0.0032482793804713803, "b:realmlp_te_cnt": 0.000847393585301589, "b:realmlp_te_s2": 0.0004964361210193442, "b:realmlp_v1": 0.0064450941750321756, "b:xgb_dig_d2": -0.001615855509527677, "b:xgb_raw_d3": 0.00433269057718474, "b:xgb_te_d2": -0.0033627181678075714}, "careful": {"s:megayak_5f_lm": -0.08450474886996073, "s:masaya_gpt2": -0.04902561986411483, "s:lm7_5fold": -0.016891411336523863, "s:gpt2_seed2026": 0.014547738041531408, "s:tabpfn_slimypunk_tp1": -0.0123457470481866, "s:sadam_tabpfn_100k_tp3": -0.015910384586476225, "s:yekenot_realmlp": -0.003253325303861341, "s:alex_glm_gen": 0.002612752988313417, "s:daoviet_gpdvae": 0.04685747302679586, "s:gbn_bag5_w512": 0.0043515522989154235, "s:sameer_glm10f": 0.015478953160303576, "s:sadam_v6_gb_xgb": -0.028037375140335462, "s:sadam_v6_gb_lgb": 0.019533386719025193, "s:sadam_v4_cat": 0.00589878484038633, "s:gb_bag5": 0.02217844040888381, "s:megayak_shallow_tree": 0.10122717023890326, "wide": 0.08081797187087163, "b:ebm_all_bag": -0.03128848896508937, "b:ebm_dig": 0.002420259254897357, "b:ebm_dig_bag": -0.006646622819934664, "b:ebm_dig_r7": 0.010794306141448952, "b:ebm_diglead_bag": -0.0117042245417344, "b:ebm_diglead_i25": -0.008918615668229131, "b:ebm_diglead_r11": 0.017495433781515475, "b:ebm_diglead_r13": 0.019436209350096583, "b:ebm_i25_bag": 0.008522596080523918, "b:ebm_i25_bag4": 0.016476990933893144, "b:ebm_lead4_bag": -0.01130787311582097, "b:gb_bag10": -0.013583645440115318, "b:gb_bag5": 0.016684891246275074, "b:gb_l3": -0.019083828087590973, "b:gb_ohk": -0.006177878317621568, "b:gbc_bag3": -0.007269120619879105, "b:gbe_bag3": 0.004681311487245302, "b:gbn_bag5_w128": -0.005040768827170892, "b:gbn_bag5_w512": 0.023637688632069328, "b:gbr_bag3": -0.00024830818253431273, "b:gbx_bag3": -0.006400911249302103, "b:gbx_bag5": 0.0036937689629500865, "b:glm_dig": -3.7306124542630744e-05, "b:glm_dl": -0.007156160004066149, "b:glm_dl_bag": 0.008441014574504154, "b:glm_dl_s50": -0.01334934207158471, "b:glm_dl_s50_bag": 0.001844211766019183, "b:glm_ohk": -0.00562390612264448, "b:glm_ohk2": 0.008907953791620765, "b:glm_ohk2_bag": -0.015300133811741116, "b:glm_ohk_c01": -0.005892186175925242, "b:glm_ohkl": -0.00010117004854648993, "b:glm_ohkl_bag": 0.008342574170709143, "b:glm_sb": -0.003455433101783884, "b:glm_te": -0.0018554925449663707, "b:lgb_dig_l4": 0.012788668778136636, "b:lgb_dig_l7": 0.006133544143597403, "b:lgb_dig_s100": -0.01581131868317444, "b:lgb_dig_s100_bag": 0.005976683240784171, "b:lgb_dig_s100_bag5": 0.003272572328457066, "b:lgb_dig_s100_l3": 0.011358913543846424, "b:lgb_dig_s100_l3_bag": -0.007671455946372908, "b:lgb_diglead_bag": 0.0026446221720072436, "b:lgb_diglead_bag5": -0.014415491763205732, "b:lgb_diglead_l2": -0.009879458752685616, "b:lgb_diglead_l3": 0.015081200121801388, "b:lgb_diglead_l3_bag": -0.01231464394864841, "b:lgb_diglead_l4": 0.0027104104670708385, "b:lgb_dk_l4": 0.002097897271309318, "b:lgb_raw_l31": 0.0010331280960220208, "b:lgb_raw_l4": -0.0013378069288362123, "b:lgb_raw_l4_cnt": -0.013507197348019013, "b:lgb_rawdig_l4": 0.0028805514532224763, "b:lgb_te_l4": 0.008125965905514013, "b:lgb_te_l4_s100": 0.0030857860977705903, "b:lgb_te_l4_s300": 0.0022465238467168505, "b:lgb_te_l4_s5": -0.0007517378836866291, "b:realmlp_allcat": -0.00013519997284003595, "b:realmlp_cnt": 0.0031564480735426013, "b:realmlp_dig": -0.009290279722601536, "b:realmlp_dig_s100": 0.005582584293801132, "b:realmlp_dig_s2": 0.0018383277487227473, "b:realmlp_rawdig": -0.009112515110809292, "b:realmlp_te": 0.003079185734258433, "b:realmlp_te_cnt": -0.00037244665110067394, "b:realmlp_te_s2": -0.0021716481344977163, "b:realmlp_v1": 0.005001493601478614, "b:xgb_dig_d2": -0.004210955412618814, "b:xgb_raw_d3": 0.004254033686957146, "b:xgb_te_d2": -0.0023974804930285173}}'

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md(f"""# {TITLE}

Playground Series S6E11: predict the probability that a card transaction is fraud.
A tutorial that builds a strong tabular ensemble step by step and shows, with a measurement each time, *why* each step is there:

0. **The problem and the data**: what one row is, every column, printed rows.
1. **The metric**: ROC AUC, and what it does and does not reward (by counterexample, not by definition).
2. **EDA**: class balance, which features separate fraud, train vs test.
3. **The ensemble**: in-fold target encoding (with a leak you can measure), shallow trees (with the curve that justifies them),
   a GLM and two neural nets, a small honest tuning run, and a blend that is scored on rows it did not fit.
4. **How this notebook got here**: every version's leaderboard score, and the ideas that did not work, with their numbers.

Every number below is computed in this notebook, except the per-version history in section 4.
All model predictions are published as a dataset ([s6e11-oof-predictions](https://www.kaggle.com/datasets/alexchilton/s6e11-oof-predictions)),
so you can blend them in your own notebook without training. If it helps you, an upvote is appreciated.

**Which file has which score.** `submission.csv` is our own stack, checked on held-out rows: public LB **0.95260**.
`submission_a370_wide10.csv` (v28) is 90% Aman Atar's public 0.95370 file and 10% our stack: public LB **0.95367**, the score in the title. Most of that file is another author's work; our own stack is the 0.95260.
`submission_p12_ours.csv` is 65% goodpjw2008's Part 12 wide stack and 35% our stack: public LB **0.95305** (section 3, v26).
Section v27 replays the public "ladder" files (0.95330 to 0.95347) on held-out rows, to see what a step picked on the public board is worth on rows it was not picked on.
Section v28 takes the public 0.95370 file apart into 16 public models and scores its mix on labelled rows (0.95186, against 0.95257 for our stack), adds two more language-model seeds to the pool, and runs two checks on every pool member.
The newest section (v29) rebuilds the public 0.95388 file byte for byte from its published weights and scores it on labelled rows (0.95246, in-sample, against 0.95257 for our held-out stack), and applies the same 87-member move to our own stack.
"""),
code("%pip install -q pytabkit"),
code("""
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=0.9)

KAGGLE = Path('/kaggle/input').exists()
DATA = next(Path('/kaggle/input').rglob('train.csv')).parent if KAGGLE else Path('../../data')   # the mount path differs between images
OUT = Path('/kaggle/working') if KAGGLE else Path('../../out')
OUT.mkdir(exist_ok=True)
THREADS = os.cpu_count()
import torch
GPU = torch.cuda.is_available()
train = pd.read_csv(DATA / 'train.csv'); test = pd.read_csv(DATA / 'test.csv')
TARGET = 'is_fraud'
print(f'train {train.shape}, test {test.shape}, cores {THREADS}')
"""),
md("""## 0. The problem and the data

One row is one card transaction. `is_fraud` is 1 when it was fraudulent.
The data is synthetic, generated from the [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) dataset;
distributions are close to, but not the same as, the original.
"""),
code("""
desc = {
 'amount_usd': 'transaction amount, USD', 'hours_since_last_txn': 'hours since the card was last used',
 'txn_count_last_24h': 'transactions on the card in the last 24 h', 'distance_from_home_km': 'distance from the cardholder\\'s home',
 'card_age_months': 'age of the card', 'customer_age': 'age of the cardholder', 'account_balance_usd': 'account balance, USD',
 'cvv_retry_count': 'failed CVV attempts', 'velocity_score': 'spending-velocity score', 'time_of_day_hour': 'hour, 0-23',
 'day_of_week': 'day, 0-6', 'merchant_risk_score': 'merchant risk score, 0-100', 'prior_disputes': 'earlier disputes on the account',
 'merchant_category': 'merchant type', 'card_type': 'card network', 'auth_method': 'how the payment was authenticated',
 'channel': 'POS, online, in-app, contactless, ATM', 'device_type': 'device used', 'is_foreign_transaction': 'foreign transaction',
 'is_new_merchant': 'first time at this merchant', 'used_vpn': 'VPN detected', 'ip_country_mismatch': 'IP country differs from card country',
 'billing_shipping_mismatch': 'billing and shipping address differ', 'is_ai_generated_scam_attempt': 'flagged as an AI-generated scam',
}
cols = pd.DataFrame([dict(column=c, dtype=str(train[c].dtype), unique=train[c].nunique(), example=train[c].iloc[0], meaning=desc[c])
                     for c in train.columns if c in desc])
display(cols)
display(train.head(5))
"""),
md("""## 1. The metric: ROC AUC

The score is the area under the ROC curve between your probabilities and the true labels.
The useful way to read it: **AUC is the chance that a random fraud gets a higher score than a random legitimate transaction.**
Only the *order* of your predictions matters. Three consequences, each shown below with a computed example:
"""),
code("""
rng = np.random.default_rng(0)
y_toy = (rng.random(10_000) < 0.0167).astype(int)                 # same fraud rate as train
good = y_toy * 1.0 + rng.normal(0, 0.6, y_toy.size)             # a decent ranking
print(f"1. all-zero predictions: accuracy {np.mean(y_toy == 0):.3f}, AUC {roc_auc_score(y_toy, np.zeros(y_toy.size)):.3f}  (98% accurate, worthless)")
print(f"2. any monotone transform keeps AUC: raw {roc_auc_score(y_toy, good):.4f}, exp() {roc_auc_score(y_toy, np.exp(good)):.4f}, "
      f"rank {roc_auc_score(y_toy, pd.Series(good).rank()):.4f}, 0.8 + x/1e6 {roc_auc_score(y_toy, 0.8 + good / 1e6):.4f}")
print(f"3. calibration is free: scores squeezed into [0.80, 0.81] (useless as probabilities) score the same AUC")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s, lab in [(good, 'decent model'), (rng.normal(size=y_toy.size), 'random')]:
    f, t, _ = roc_curve(y_toy, s); ax[0].plot(f, t, label=f'{lab} (AUC {roc_auc_score(y_toy, s):.3f})')
ax[0].plot([0, 1], [0, 1], 'k:', lw=1); ax[0].set_xlabel('false positive rate'); ax[0].set_ylabel('true positive rate'); ax[0].legend(); ax[0].set_title('ROC curve')
ax[1].hist(good[y_toy == 0], bins=60, alpha=.6, density=True, label='legit'); ax[1].hist(good[y_toy == 1], bins=60, alpha=.6, density=True, label='fraud')
ax[1].set_title('AUC = P(fraud score > legit score)'); ax[1].legend(); plt.tight_layout(); plt.show()
"""),
md("""What this means for the competition:

- Class imbalance (1.7% fraud) does not need resampling or class weights for AUC; a model that ranks well is enough.
- Blending is easy: average **ranks** (or probabilities) of several models; scale differences between models do not matter.
- Small AUC gains are real gains here: with 300k test rows, a difference of 0.0005 is many correctly re-ordered pairs, but check it in CV, not on the public board alone.
"""),
md("## 2. EDA\n\n### Class balance and the numeric features by class"),
code("""
y = train[TARGET].astype(int)
print(f"fraud rate {y.mean():.4f} ({y.sum():,} of {len(y):,}); missing values: {int(train.isna().sum().sum())} in train, {int(test.isna().sum().sum())} in test")
num = [c for c in train.columns if train[c].dtype.kind in 'if' and c not in ('id', TARGET)]
cat = train.select_dtypes('object').columns.tolist()
boo = train.select_dtypes('bool').columns.tolist()
auc1 = {c: roc_auc_score(y, train[c]) for c in num}
auc1 = pd.Series({c: max(a, 1 - a) for c, a in auc1.items()}).sort_values(ascending=False)
print('single-feature AUC (how well each column alone orders fraud):'); print(auc1.round(3).to_string())
fig, axs = plt.subplots(3, 5, figsize=(16, 8)); axs = axs.ravel()
for ax, c in zip(axs, auc1.index):
    lo, hi = train[c].quantile([0.005, 0.995])
    for lab, g in train.groupby(TARGET):
        ax.hist(g[c].clip(lo, hi), bins=40, density=True, alpha=.55, label='fraud' if lab else 'legit')
    ax.set_title(f'{c}  (AUC {auc1[c]:.3f})', fontsize=8)
for ax in axs[len(auc1):]: ax.axis('off')
axs[0].legend(); plt.tight_layout(); plt.show()
"""),
md("### Fraud rate by category and by flag"),
code("""
fig, axs = plt.subplots(2, 3, figsize=(16, 7)); axs = axs.ravel()
for ax, c in zip(axs, cat):
    r = train.groupby(c)[TARGET].agg(['mean', 'size']).sort_values('mean')
    ax.barh(r.index.astype(str), r['mean']); ax.axvline(y.mean(), color='k', ls=':', lw=1); ax.set_title(f'fraud rate by {c}', fontsize=9)
r = pd.DataFrame({c: [train.loc[train[c], TARGET].mean(), train.loc[~train[c], TARGET].mean(), train[c].mean()] for c in boo},
                 index=['rate if True', 'rate if False', 'share True']).T.sort_values('rate if True')
axs[-1].barh(r.index, r['rate if True'], label='flag True'); axs[-1].barh(r.index, r['rate if False'], label='flag False', alpha=.6)
axs[-1].axvline(y.mean(), color='k', ls=':', lw=1); axs[-1].set_title('fraud rate by flag', fontsize=9); axs[-1].legend(fontsize=7)
plt.tight_layout(); plt.show()
display(r.round(4))
"""),
md("""### Train vs test: are they the same distribution?

An adversarial check: train a quick model to tell train rows from test rows.
AUC near 0.5 means the two sets look alike, so CV on train should predict the leaderboard.
"""),
code("""
import lightgbm as lgb
feats = [c for c in train.columns if c not in ('id', TARGET)]
def prep(df):
    d = df[feats].copy()
    d[boo] = d[boo].astype(int)
    for c in cat: d[c] = pd.Categorical(d[c], categories=sorted(train[c].unique()))
    return d
X, XT = prep(train), prep(test)
adv = pd.concat([X, XT]); adv_y = np.r_[np.zeros(len(X)), np.ones(len(XT))]
a_oof = np.zeros(len(adv))
for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=0).split(adv, adv_y):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, n_jobs=THREADS, verbose=-1).fit(adv.iloc[tr_i], adv_y[tr_i])
    a_oof[va_i] = m.predict_proba(adv.iloc[va_i])[:, 1]
print(f'adversarial AUC train vs test: {roc_auc_score(adv_y, a_oof):.4f}')
"""),
md("""## 3. The ensemble

Gradient-boosted trees (LightGBM, XGBoost, CatBoost), a spline GLM and two neural nets (RealMLP, TabM), all trained on the same 5 stratified folds,
so every model's out-of-fold predictions line up row for row and can be blended.

- **LightGBM / XGBoost / CatBoost**: gradient-boosted trees, native categorical handling, early stopping on the fold's validation AUC.
- **RealMLP** ([pytabkit](https://github.com/dholzmueller/pytabkit), Holzmüller et al., NeurIPS 2024): an MLP with tuned defaults for tables
  (robust scaling, periodic numeric embeddings, a learned per-feature scale). It makes different errors from trees, which is what a blend needs.
- **Target encoding inside each fold** (sklearn `TargetEncoder`, cross-fitted, so a row never sees its own label): every column with >= 100 rows per value
  (numeric ones per exact value), integer-rounded keys for the four continuous columns, and pairs among the ten columns with the strongest per-value signal.
  Not all 276 pairs: on three earlier playgrounds all-pair crosses measured -0.0036, -0.0004 and +0.0002.
- **Shallow trees** (4-leaf LightGBM twice, depth-1 and depth-2 XGBoost, depth-3 CatBoost, on the encoded and the raw inputs): the signal is close to additive; the top public notebooks measured +0.0019 over 31 leaves, and in v5 the blend kept only the shallow members.
- **TabM** (a batch ensemble of 32 MLPs, GPU) next to RealMLP.
- **A spline GLM** (logistic regression on splines and one-hots; design from sidhaarthshree and goodpjw2008).
- **RealMLP with tuned settings** (yekenot's S6E10 parameters, used by ravi20076 here), on the encoded inputs.
- **Decimal digits**: the first decimal digit of five numeric columns; the synthesiser leaves a pattern there that range splits cannot express.
- **The original 20k rows as a lookup**: per-value fraud rates in the original data and a small model trained only on it, as features (not as extra rows).
- LightGBM runs twice (raw / engineered); the other models use the engineered set only if it gains at least 0.0002 OOF AUC, so the notebook decides.
"""),
md("""### The pipeline in one picture

Red marks the one place a label can leak: anything fitted on labels (the target encoder) must be fitted inside the fold's training rows.
"""),
code("""
fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis('off'); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)
def box(x, y, w, h, text, fc='#eef3fb', ec='#4a6fa5'):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5)); ax.text(x + w / 2, y + h / 2, text.replace('|', chr(10)), ha='center', va='center', fontsize=9)
def arrow(x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=1.3, color='#555'))
box(0.1, 1.6, 1.7, 1.0, 'train 700k|+ test 300k')
box(2.3, 2.7, 2.0, 1.0, 'fold k: training|rows (4/5)')
box(2.3, 0.5, 2.0, 1.0, 'fold k: validation|rows (1/5)')
box(4.9, 2.7, 2.2, 1.0, 'target encoder|FITTED HERE ONLY', fc='#fdecea', ec='#c0392b')
box(4.9, 0.5, 2.2, 1.0, 'encoder applied|(no labels used)')
box(7.7, 1.6, 2.0, 1.0, 'every model|(trees, GLM, MLPs)')
box(10.2, 2.7, 2.6, 1.0, 'out-of-fold predictions|-> blend weights')
box(10.2, 0.5, 2.6, 1.0, 'test predictions|(mean of 5 folds)')
for a in [(1.8, 2.3, 2.3, 3.2), (1.8, 1.9, 2.3, 1.0), (4.3, 3.2, 4.9, 3.2), (4.3, 1.0, 4.9, 1.0), (6.0, 2.7, 6.0, 1.5),
          (7.1, 3.2, 7.7, 2.4), (7.1, 1.0, 7.7, 1.8), (9.7, 2.3, 10.2, 3.2), (9.7, 1.9, 10.2, 1.0)]:
    arrow(*a)
ax.set_title('Every model sees the same 5 folds; the encoder never sees a validation label', fontsize=11); plt.show()
"""),
md("### Features: targeted encodings, decimal digits, the original data as a lookup"),
code("""
from itertools import combinations
from sklearn.preprocessing import TargetEncoder
T_START = time.time()
both = pd.concat([train[feats], test[feats]], ignore_index=True)
cont = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd']
P = {}
for c in feats:                                      # every column, each exact value a category; the encoder's smoothing handles rare values
    codes, uniq = pd.factorize(both[c].astype(str))
    if len(uniq) <= len(both) // 2:
        P[c] = codes.astype('int32')
for c in cont:                                       # the continuous ones pooled on rounded keys instead, two levels
    P[f'{c}_int'] = pd.factorize(np.round(both[c]).astype(int))[0].astype('int32')
    P[f'{c}_r10'] = pd.factorize(np.round(both[c] / 10).astype(int))[0].astype('int32')
for c in ['merchant_risk_score', 'velocity_score']:  # exact value is a single key above; the integer level pools neighbours
    P[f'{c}_int'] = pd.factorize(np.round(both[c]).astype(int))[0].astype('int32')
SINGLES = list(P)
# pairs only among the ten columns with the strongest per-level signal (the cron EDA notebook's signal ratios)
TOP = ['cvv_retry_count', 'ip_country_mismatch', 'is_foreign_transaction', 'is_new_merchant', 'billing_shipping_mismatch',
       'auth_method', 'used_vpn', 'is_ai_generated_scam_attempt', 'merchant_category', 'txn_count_last_24h']
for a, b in combinations(TOP, 2):
    P[f'{a}__{b}'] = pd.factorize(both[a].astype(str) + '_' + both[b].astype(str))[0].astype('int32')
P = pd.DataFrame(P); ENC = list(P.columns)
PT, P = P.iloc[len(train):].reset_index(drop=True), P.iloc[:len(train)].reset_index(drop=True)
print(f'target-encoded: {len(SINGLES)} single keys + {len(ENC) - len(SINGLES)} pairs among the top 10')

# first decimal digit: the synthesiser leaves a pattern there that no range split can express (a modulo condition)
DIGIT_COLS = ['merchant_risk_score', 'velocity_score', 'hours_since_last_txn', 'amount_usd', 'distance_from_home_km']
digit = lambda d: pd.DataFrame({f'd1_{c}': (np.floor(d[c].values * 10 + 1e-6) % 10).astype('int8') for c in DIGIT_COLS})
D, DT = digit(train), digit(test)
display(pd.DataFrame({c: train.groupby(D[f'd1_{c}'].values)[TARGET].mean() for c in DIGIT_COLS}).round(4).T)

# the original 20k rows as a lookup, not as extra training rows: per-value fraud rates there, and a model trained only on them
orig_path = next(iter(Path('/kaggle/input').rglob('credit_card_fraud_2026.csv')), None) if KAGGLE else Path('../../data/orig/credit_card_fraud_2026.csv')
G, GT = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
if orig_path is not None and Path(orig_path).exists():
    orig = pd.read_csv(orig_path); oy = orig[TARGET].astype(int); prior = oy.mean()
    print(f'original data: {len(orig):,} rows, fraud rate {prior:.4f}')
    for c in ['merchant_risk_score', 'velocity_score', 'customer_age'] + cat + boo:
        m = orig.groupby(orig[c].astype(str))[TARGET].agg(['sum', 'size'])
        rate = (m['sum'] + 20 * prior) / (m['size'] + 20)            # smoothed toward the overall rate
        G[f'orig_{c}'] = train[c].astype(str).map(rate).fillna(prior).astype('float32').values
        GT[f'orig_{c}'] = test[c].astype(str).map(rate).fillna(prior).astype('float32').values
    om = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=THREADS, verbose=-1).fit(prep(orig), oy)
    G['orig_model'] = om.predict_proba(X)[:, 1].astype('float32'); GT['orig_model'] = om.predict_proba(XT)[:, 1].astype('float32')
else:
    print('original data not attached: no lookup features')
print(f'lookup features from the original data: {G.shape[1]}')

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
CACHE = {}
def frames(k, mode):
    \"\"\"(train, valid, test) inputs for fold k. The encodings are fitted on the fold's training rows only (cross-fitted 5 ways
    inside them), so no row sees its own label; computed once per fold and reused by every model.\"\"\"
    tr_i, va_i = FOLDS[k]
    Xa, Xb, Xt = X.iloc[tr_i].reset_index(drop=True), X.iloc[va_i].reset_index(drop=True), XT.reset_index(drop=True)
    if mode == 'raw':
        return Xa, Xb, Xt
    if k not in CACHE:
        enc = TargetEncoder(target_type='binary', cv=5, random_state=0)
        te = lambda arr: pd.DataFrame(arr, columns=['te_' + c for c in ENC]).astype('float32')
        a = te(enc.fit_transform(P.iloc[tr_i], y.iloc[tr_i])); b = te(enc.transform(P.iloc[va_i])); t = te(enc.transform(PT))
        g = lambda df, idx: df.iloc[idx].reset_index(drop=True) if idx is not None else df.reset_index(drop=True)
        CACHE[k] = (pd.concat([g(D, tr_i), g(G, tr_i), a], axis=1), pd.concat([g(D, va_i), g(G, va_i), b], axis=1),
                    pd.concat([g(DT, None), g(GT, None), t], axis=1))
    a, b, t = CACHE[k]
    return pd.concat([Xa, a], axis=1), pd.concat([Xb, b], axis=1), pd.concat([Xt, t], axis=1)
"""),
md("""### Why the encoder must be fitted inside the fold: a leak you can measure

Target encoding replaces a value by the fraud rate of the rows that share it. If those rates are computed on **all** rows before the CV split,
a validation row's own label is inside its feature. The CV score then measures memory, not prediction.

Below, 40k rows are put aside as a stand-in for the test set and never used to fit anything. On the other 200k rows the same encodings
are built two ways, and each is scored twice: by 3-fold CV, and on the untouched 40k.
If CV and hold-out disagree, the CV is lying.
"""),
code("""
rs = np.random.RandomState(0); idx = rs.permutation(len(X))[:240_000]
hold, dev = idx[:len(idx) // 6], idx[len(idx) // 6:]          # 40k / 200k
cols = SINGLES
lgb4 = lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.1, num_leaves=4, n_jobs=THREADS, verbose=-1)
leak = {}
for how in ['fitted on all rows first (leaky)', 'fitted inside each fold (honest)']:
    leaky = how.endswith('(leaky)'); cv = []
    for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=0).split(dev, y.iloc[dev]):
        a_i, b_i = dev[tr_i], dev[va_i]
        enc = TargetEncoder(target_type='binary', random_state=0)
        if leaky:                                          # fit + transform on the same rows: no cross-fitting, validation labels included
            enc.fit(P.iloc[dev][cols], y.iloc[dev]); A, B = enc.transform(P.iloc[a_i][cols]), enc.transform(P.iloc[b_i][cols])
        else:                                              # fit_transform cross-fits inside the training rows; validation only transformed
            A = enc.fit_transform(P.iloc[a_i][cols], y.iloc[a_i]); B = enc.transform(P.iloc[b_i][cols])
        cv.append(roc_auc_score(y.iloc[b_i], lgb4().fit(A, y.iloc[a_i]).predict_proba(B)[:, 1]))
    enc = TargetEncoder(target_type='binary', random_state=0)
    A = (enc.fit(P.iloc[dev][cols], y.iloc[dev]).transform(P.iloc[dev][cols]) if leaky else enc.fit_transform(P.iloc[dev][cols], y.iloc[dev]))
    hold_auc = roc_auc_score(y.iloc[hold], lgb4().fit(A, y.iloc[dev]).predict_proba(enc.transform(P.iloc[hold][cols]))[:, 1])
    leak[how] = (np.mean(cv), hold_auc)
L = pd.DataFrame(leak, index=['3-fold CV AUC', 'hold-out AUC']).T; display(L.round(5))
ax = L.plot.bar(figsize=(8, 3.5), rot=0, color=['#9bb7d4', '#4a6fa5'], ylim=(L.values.min() - 0.01, min(1.0, L.values.max() + 0.01)))
ax.set_title('Leaky encoding: CV promises more than the hold-out delivers'); ax.set_ylabel('AUC'); plt.show()
"""),
code("""
oof, pred, secs = {}, {}, {}
BUDGET_H = 9.0                                         # a model that would start after this is skipped (Kaggle stops at 12 h)

# Every member's out-of-fold and test predictions are published as the dataset alexchilton/s6e11-oof-predictions, keyed by a
# fingerprint of (member name, input set, the fitting code and the settings it reads, a checksum of fold 0's exact inputs, the folds).
# A member whose fingerprint is already there is loaded instead of trained; change its code, its settings or the features
# and the fingerprint changes, so it retrains. Attach the dataset to your own notebook to blend these members without training them.
import hashlib, json, types
PRED_CACHE = next((f.parent for f in Path('/kaggle/input').rglob('oof_manifest.json')), None) if KAGGLE else None
MANIFEST = json.loads((PRED_CACHE / 'oof_manifest.json').read_text()) if PRED_CACHE else {}
print('prediction cache:', f'{PRED_CACHE} ({len(MANIFEST)} members)' if PRED_CACHE else 'none attached, every member trains')
__FP_CORE__
import inspect
def fingerprint(name, fit_predict, mode):
    g = globals()
    funcs = {k: inspect.getsource(v) for k, v in g.items()
             if isinstance(v, types.FunctionType) and v.__module__ == '__main__' and v.__name__ != '<lambda>' and not k.startswith('_')}
    settings = {k: v for k, v in g.items() if type(v) in (int, float, str, bool, dict) and not k.startswith('_')}
    try:
        feature_src = fp_feature_cell(get_ipython().user_ns['In'])          # the cell as it was run
    except Exception:                                                     # outside IPython: no cache, a name-based key
        return 'nofp_' + ''.join(ch if ch.isalnum() else '_' for ch in f'{name}_{mode}')
    return fp_compute(name, mode, fit_predict.__name__, funcs, settings, feature_src)

def run(name, fit_predict, mode=None):
    mode = mode or MODE
    if (time.time() - T_START) / 3600 > BUDGET_H:
        print(f'{name}: skipped, {BUDGET_H} h budget used'); return
    fp = fingerprint(name, fit_predict, mode)
    if fp in MANIFEST and (PRED_CACHE / f'oof_{fp}.npy').exists():
        oof[name], pred[name] = np.load(PRED_CACHE / f'oof_{fp}.npy'), np.load(PRED_CACHE / f'test_{fp}.npy'); secs[name] = MANIFEST[fp]['minutes'] * 60
        _publish(fp, name, mode)
        print(f'{name}: OOF AUC {roc_auc_score(y, oof[name]):.5f}  (loaded from the dataset, trained in {MANIFEST[fp]["version"]})'); return
    t0 = time.time(); o = np.zeros(len(X)); p = np.zeros(len(XT))
    for k, (tr_i, va_i) in enumerate(FOLDS):
        Xa, Xb, Xt = frames(k, mode)
        vo, tp = fit_predict(Xa, y.iloc[tr_i].values, Xb, y.iloc[va_i].values, Xt)
        o[va_i] = vo; p += tp / len(FOLDS)
        print(f'  {name} fold {k}: AUC {roc_auc_score(y.iloc[va_i], vo):.5f}  ({Xa.shape[1]} inputs)', flush=True)
    oof[name], pred[name], secs[name] = o, p, time.time() - t0
    _publish(fp, name, mode, trained=True)
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}  ({secs[name] / 60:.1f} min)')

OUT_MANIFEST = {}
def _publish(fp, name, mode, trained=False):         # this run's output holds every member, trained or loaded, for the next dataset version
    np.save(OUT / f'oof_{fp}.npy', oof[name]); np.save(OUT / f'test_{fp}.npy', pred[name])
    OUT_MANIFEST[fp] = dict(name=name, inputs=mode, oof_auc=round(float(roc_auc_score(y, oof[name])), 5), minutes=round(secs[name] / 60, 1),
                            version=VERSION if trained else MANIFEST[fp]['version'])
    (OUT / 'oof_manifest.json').write_text(json.dumps(OUT_MANIFEST, indent=1))
"""),
code("""
def lgbm(Xa, ya, Xb, yb, Xt):
    m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM raw', lgbm, 'raw')
run('LightGBM +features', lgbm, 'fe')
# the other three models use the engineered set only if it beat raw by more than run-to-run noise
gain = roc_auc_score(y, oof['LightGBM +features']) - roc_auc_score(y, oof['LightGBM raw'])
MODE = 'fe' if gain >= 0.0002 else 'raw'
print(f'engineered features: {gain:+.5f} OOF AUC for LightGBM -> the other models use {MODE!r}')

def lgbm_shallow(Xa, ya, Xb, yb, Xt):                 # 4 leaves: the signal is close to additive, so tiny trees generalise better
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves', lgbm_shallow, 'fe')
run('LightGBM 4 leaves raw', lgbm_shallow, 'raw')
def lgbm_shallow2(Xa, ya, Xb, yb, Xt):                # the same tree size, another seed and a sparser column sample
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=200, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.3, reg_lambda=5.0, random_state=1, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves b', lgbm_shallow2, 'fe')
"""),
md("""### How big should a tree be? Measure it

One fold, the same inputs, only `num_leaves` changes. A deep tree can express interactions between many columns;
if the score falls as trees grow, those interactions are mostly noise here, and the target is close to a sum of per-column effects.
"""),
code("""
Xa0, Xb0, _ = frames(0, MODE); ya0, yb0 = y.iloc[FOLDS[0][0]].values, y.iloc[FOLDS[0][1]].values
sizes, curve = [2, 4, 8, 16, 32, 64], []
for L in sizes:
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.1, num_leaves=L, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, n_jobs=THREADS, verbose=-1)
    m.fit(Xa0, ya0, eval_set=[(Xb0, yb0)], eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    curve.append(m.best_score_['valid_0']['auc'])
fig, ax = plt.subplots(figsize=(7, 3.5)); ax.plot(sizes, curve, 'o-', color='#4a6fa5'); ax.set_xscale('log', base=2); ax.set_xticks(sizes, sizes)
ax.set_xlabel('leaves per tree'); ax.set_ylabel('validation AUC (fold 0)'); ax.set_title('Validation AUC against tree size'); plt.show()
print(dict(zip(sizes, np.round(curve, 5))))
"""),
md("""### What the model learned, one column at a time

Trees of depth 1 use one column per tree, so the model is a sum of one curve per column (a *shape function*).
XGBoost's per-feature contributions (`pred_contribs`) then are exactly those curves. Steps and bumps are what the encodings and digits try to capture.
"""),
code("""
import xgboost as xgb
Xa1, Xb1, _ = frames(0, 'raw')
m1 = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.1, max_depth=1, tree_method='hist', device='cuda' if GPU else 'cpu',
                       enable_categorical=True, eval_metric='auc', early_stopping_rounds=200, n_jobs=THREADS)
m1.fit(Xa1, ya0, eval_set=[(Xb1, yb0)], verbose=False); m1.set_params(device='cpu')
smp = Xb1.sample(min(20000, len(Xb1)), random_state=0)
contrib = pd.DataFrame(m1.get_booster().predict(xgb.DMatrix(smp, enable_categorical=True), pred_contribs=True)[:, :-1], columns=smp.columns)
top = contrib.abs().mean().sort_values(ascending=False).index[:12]
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
for ax, c in zip(axes.ravel(), top):
    if smp[c].dtype.name == 'category' or smp[c].nunique() <= 12:
        g = contrib[c].groupby(smp[c].astype(str).values).mean().sort_values(); ax.barh(g.index, g.values, color='#4a6fa5')
    else:
        ax.scatter(smp[c], contrib[c], s=2, alpha=0.3, color='#4a6fa5')
    ax.set_title(c, fontsize=9); ax.axhline(0, color='#999', lw=0.8)
fig.suptitle('Shape functions of a depth-1 XGBoost (log-odds contribution per column)', y=1.0); plt.tight_layout(); plt.show()
print(f'depth-1 XGBoost: {m1.best_iteration + 1} trees, fold-0 validation AUC {roc_auc_score(yb0, m1.predict_proba(Xb1)[:, 1]):.5f}')
"""),
md("""### Tuning without fooling yourself: small Optuna searches

Each search uses one fold and a 250k-row sample of its training rows, with a fast learning rate, so 30 trials cost minutes, not hours.
The found settings then train a new member at the usual learning rate on all five folds.
Tuning on fold 0's validation rows makes that fold slightly optimistic; the other four folds are untouched, which keeps the OOF score honest enough.
The plots show what a search is for: whether more trials still help (history), and which settings matter at all (importance).

A search is cached in the dataset like the members, keyed by its objective's code, so the next version reloads its trials instead of searching again.

**LightGBM was searched in v8** (30 trials, 28 minutes on 4 CPU cores): the best setting is num_leaves = 2, two-leaf trees, even smaller than our hand-picked 4.
At the full learning rate that member scores OOF 0.94839, the same as the hand-picked 4-leaf LightGBM (0.94838), and took 62 minutes.
The hill climb gives it weight 0 and the stacker 0.142; the leaderboard moved from 0.94982 (v6) to 0.94981 (v8).
So the search confirmed the shallow-tree finding and found nothing beyond it. Its settings are pinned below so it is not searched again.
"""),
code("""
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
optuna.logging.set_verbosity(optuna.logging.WARNING)
sub_i = np.random.RandomState(0).permutation(len(Xa0))[:250_000]
def cached_search(tag, objective, n_trials):
    \"\"\"An Optuna study for one member, loaded from the dataset when the same objective was searched before, else run and saved.\"\"\"
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    key = hashlib.sha1((tag + _tree(inspect.getsource(objective)) + repr(len(sub_i))).encode()).hexdigest()[:12]
    saved = PRED_CACHE / f'optuna_{key}.json' if PRED_CACHE else None
    if saved is not None and saved.exists():
        D = optuna.distributions
        for t in json.loads(saved.read_text()):
            study.add_trial(optuna.trial.create_trial(params=t['params'], value=t['value'],
                                                      distributions={k: D.json_to_distribution(v) for k, v in t['dists'].items()}))
        print(f'{tag}: search loaded from the dataset, {len(study.trials)} trials')
    else:
        log_trial = lambda s, t: print(f'  {tag} trial {t.number:2d}: AUC {t.value:.5f}  ({t.duration.total_seconds():.0f} s)', flush=True)
        study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])   # a fixed trial count, so the result does not depend on the machine
        (OUT / f'optuna_{key}.json').write_text(json.dumps([dict(params=t.params, value=t.value, seconds=t.duration.total_seconds(),
            dists={k: optuna.distributions.distribution_to_json(d) for k, d in t.distributions.items()}) for t in study.trials]))
    print(f'{tag}: best fold-0 AUC {study.best_value:.5f}', study.best_params)
    plot_optimization_history(study, target_name='fold-0 AUC'); plt.title(f'{tag}: search history'); plt.show()
    plot_param_importances(study); plt.title(f'{tag}: which settings matter'); plt.show()
    return study

# LightGBM: searched in v8 (30 trials, best fold-0 AUC 0.94568); the result is pinned, so the member below loads from the dataset
BEST = {'num_leaves': 2, 'min_child_samples': 455, 'colsample_bytree': 0.48621093026380613, 'subsample': 0.979109441929957, 'reg_lambda': 22.84848036668211}
def lgbm_tuned(Xa, ya, Xb, yb, Xt):
    m = lgb.LGBMClassifier(n_estimators=30000, learning_rate=0.03, subsample_freq=1, n_jobs=THREADS, verbose=-1, **BEST)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(400, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM tuned', lgbm_tuned)
"""),
code("""

def lgbm_orig(Xa, ya, Xb, yb, Xt):                    # starts from the original-data model's logit and learns only the correction
    if 'orig_model' not in Xa:
        return lgbm(Xa, ya, Xb, yb, Xt)
    lg = lambda d: np.log(np.clip(d['orig_model'].values, 1e-6, 1 - 1e-6) / (1 - np.clip(d['orig_model'].values, 1e-6, 1 - 1e-6)))
    m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, init_score=lg(Xa), eval_set=[(Xb, yb)], eval_init_score=[lg(Xb)], eval_metric='auc',
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xb, raw_score=True) + lg(Xb), m.predict(Xt, raw_score=True) + lg(Xt)
run('LightGBM from original', lgbm_orig, 'fe')

# a spline GLM (design from sidhaarthshree's and goodpjw2008's notebooks): the original data looks like a logistic formula,
# so a linear model on splines + one-hots sees it differently from the trees, which is what a stack wants
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
full = pd.concat([train[feats], test[feats]], ignore_index=True)
nums = [c for c in num if c in feats]
zz = full[nums].astype(float).copy()
for c in ('amount_usd', 'account_balance_usd', 'hours_since_last_txn'):
    zz[c] = np.log1p(zz[c])
zz = StandardScaler().fit_transform(zz)
small = [c for c in nums if full[c].nunique() <= 100]
M = sparse.hstack([sparse.csr_matrix(zz), sparse.csr_matrix(full[boo].values.astype(float)),
                   OneHotEncoder(handle_unknown='ignore').fit_transform(full[cat].astype(str)),
                   sparse.csr_matrix(SplineTransformer(n_knots=8, degree=3, knots='quantile').fit_transform(zz)),
                   OneHotEncoder(handle_unknown='ignore', min_frequency=50).fit_transform(full[small])]).tocsr()
MA, MT = M[:len(train)], M[len(train):]
glm_key = 'spline_glm_' + hashlib.sha1(M.data[:200000].tobytes() + repr(M.shape).encode()).hexdigest()[:10]   # keyed by the design matrix
if glm_key in MANIFEST and (PRED_CACHE / f'oof_{glm_key}.npy').exists():
    oof['Spline GLM'], pred['Spline GLM'] = np.load(PRED_CACHE / f'oof_{glm_key}.npy'), np.load(PRED_CACHE / f'test_{glm_key}.npy')
    secs['Spline GLM'] = MANIFEST[glm_key]['minutes'] * 60; _publish(glm_key, 'Spline GLM', 'design')
    print(f"Spline GLM: OOF AUC {roc_auc_score(y, oof['Spline GLM']):.5f}  (loaded from the dataset, trained in {MANIFEST[glm_key]['version']})")
else:
    t0 = time.time(); o = np.zeros(len(X)); p = np.zeros(len(XT))
    for k, (tr_i, va_i) in enumerate(FOLDS):
        m = LogisticRegression(C=1.0, max_iter=4000, tol=1e-6).fit(MA[tr_i], y.iloc[tr_i])
        o[va_i] = m.predict_proba(MA[va_i])[:, 1]; p += m.predict_proba(MT)[:, 1] / len(FOLDS)
    oof['Spline GLM'], pred['Spline GLM'], secs['Spline GLM'] = o, p, time.time() - t0
    _publish(glm_key, 'Spline GLM', 'design', trained=True)
    print(f"Spline GLM: OOF AUC {roc_auc_score(y, o):.5f}  ({secs['Spline GLM'] / 60:.1f} min, {M.shape[1]} design columns)")
"""),
code("""
import xgboost as xgb
def xgbm(Xa, ya, Xb, yb, Xt):
    m = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc', early_stopping_rounds=200, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost', xgbm)
def xgbm_deep(Xa, ya, Xb, yb, Xt):                    # a second XGBoost: deeper, slower learning, different column sampling
    m = xgb.XGBClassifier(n_estimators=6000, learning_rate=0.02, max_depth=9, min_child_weight=20, subsample=0.7, colsample_bytree=0.35,
                          reg_lambda=5.0, tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=300, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost deep', xgbm_deep)
def xgbm_d2(Xa, ya, Xb, yb, Xt):                      # depth 2: the XGBoost version of the 4-leaf trees
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.05, max_depth=2, min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=300, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 2', xgbm_d2, 'fe')
run('XGBoost depth 2 raw', xgbm_d2, 'raw')
def xgbm_d1(Xa, ya, Xb, yb, Xt):                      # depth 1: a purely additive model, one shape function per input
    m = xgb.XGBClassifier(n_estimators=30000, learning_rate=0.1, max_depth=1, min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=500, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 1', xgbm_d1, 'fe')
"""),
code("""
from catboost import CatBoostClassifier
def catb(Xa, ya, Xb, yb, Xt):
    s = lambda d: d.assign(**{c: d[c].astype(str) for c in cat})
    m = CatBoostClassifier(iterations=CAT_ITERS, learning_rate=0.06, depth=6, eval_metric='AUC', od_type='Iter', od_wait=200,
                           thread_count=THREADS, verbose=0, cat_features=cat, random_seed=0, train_dir=str(OUT / 'catboost_info'),
                           task_type='GPU' if GPU else 'CPU')
    m.fit(s(Xa), ya, eval_set=(s(Xb), yb))
    return m.predict_proba(s(Xb))[:, 1], m.predict_proba(s(Xt))[:, 1]
run('CatBoost', catb)
def catb3(Xa, ya, Xb, yb, Xt):                         # shallow CatBoost: symmetric depth-3 trees
    s = lambda d: d.assign(**{c: d[c].astype(str) for c in cat})
    m = CatBoostClassifier(iterations=3 * CAT_ITERS, learning_rate=0.08, depth=3, eval_metric='AUC', od_type='Iter', od_wait=300,
                           thread_count=THREADS, verbose=0, cat_features=cat, random_seed=0, train_dir=str(OUT / 'catboost_info'),
                           task_type='GPU' if GPU else 'CPU')
    m.fit(s(Xa), ya, eval_set=(s(Xb), yb))
    return m.predict_proba(s(Xb))[:, 1], m.predict_proba(s(Xt))[:, 1]
run('CatBoost depth 3', catb3)
"""),
code("""
import torch
from pytabkit import RealMLP_TD_Classifier
torch.set_num_threads(THREADS)
print('RealMLP on', 'GPU' if GPU else 'CPU')
def realmlp(Xa, ya, Xb, yb, Xt):
    m = RealMLP_TD_Classifier(device='cuda' if GPU else 'cpu', random_state=0, verbosity=0, val_metric_name='1-auc_ovr',
                              n_ens=8, n_epochs=MLP_EPOCHS, batch_size=512, use_early_stopping=False, lr=0.04, wd=0.01, sq_mom=0.988,
                              lr_sched='flat_anneal', wd_sched='cos_log_15', first_layer_lr_factor=0.25, embedding_size=5,
                              max_one_hot_cat_size=2, hidden_sizes=[512, 256, 128], act='mish', p_drop=0.05, p_drop_sched='pow5',
                              plr_hidden_1=16, plr_hidden_2=8, plr_act_name='gelu', plr_lr_factor=0.1151, plr_sigma=0.75,
                              ls_eps=0.01, ls_eps_sched='sqrt_cos', add_front_scale=False, bias_init_mode='neg-uniform-dynamic-2',
                              tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize'])
    m.fit(Xa, ya, X_val=Xb, y_val=yb)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('RealMLP', realmlp, 'fe')
"""),
code("""
import torch
# TabM (Gorishniy et al., ICLR 2025): k MLPs sharing most weights, trained together as one batch ensemble;
# piecewise-linear embeddings of the numeric inputs. Early stopping on the fold's validation rows, as the trees do.
from pytabkit import TabM_D_Classifier
def tabm(Xa, ya, Xb, yb, Xt):
    m = TabM_D_Classifier(device='cuda' if GPU else 'cpu', random_state=0, verbosity=0, val_metric_name='1-auc_ovr',
                          arch_type='tabm-mini', tabm_k=32, num_emb_type='pwl', d_block=512, n_blocks=3, dropout=0.1,
                          batch_size=1024, lr=2e-3, weight_decay=3e-4, n_epochs=40, patience=6)
    m.fit(Xa, ya, X_val=Xb, y_val=yb)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('TabM', tabm, 'fe')
"""),
md("""### v9: the four "what next" items from v8

1. **Tuning the other shallow members.** The same cached search, run on XGBoost depth 2 and CatBoost depth 3.
2. **More features**: count encodings (how common a value is across train and test; no labels used), three products and ratios of the strongest numeric columns,
   and target encodings of triple crosses among the six strongest flags, fitted inside the fold like every other encoding. They feed two new members only,
   so every member trained before keeps its cached predictions.
3. **A bigger member library**: a k-nearest-neighbour model (the fraud rate among a row's 100 nearest training rows, GPU), and seed bags
   (the same depth-1 / depth-2 XGBoost trained with three seeds and averaged). Each is judged by what it adds to the stack, not by its own score.
4. **More seeds for the blend**: the stacker's held-out AUC is measured over three different outer splits, so the hill climb vs stacker choice comes with its noise.
   More folds for the members would change every out-of-fold file and retrain everything, so the 5 folds stay.
"""),
code("""
# 1. tuned XGBoost depth <= 3 and CatBoost depth <= 4, searched like LightGBM (fold 0, 250k-row sample, fast learning rate)
def xgb_objective(trial):
    prm = dict(max_depth=trial.suggest_int('max_depth', 1, 3), min_child_weight=trial.suggest_float('min_child_weight', 1, 300, log=True),
               subsample=trial.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 0.9),
               reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 100, log=True))
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.1, tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True,
                          eval_metric='auc', early_stopping_rounds=200, n_jobs=THREADS, **prm)
    m.fit(Xa0.iloc[sub_i], ya0[sub_i], eval_set=[(Xb0, yb0)], verbose=False)
    return m.best_score
XGB_BEST = cached_search('XGBoost', xgb_objective, 30).best_params
def xgbm_tuned(Xa, ya, Xb, yb, Xt):
    m = xgb.XGBClassifier(n_estimators=30000, learning_rate=0.05, tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True,
                          eval_metric='auc', early_stopping_rounds=400, n_jobs=THREADS, **XGB_BEST)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost tuned', xgbm_tuned, 'fe')

def cat_objective(trial):
    s = lambda d: d.assign(**{c: d[c].astype(str) for c in cat})
    prm = dict(depth=trial.suggest_int('depth', 2, 4), l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.1, 30, log=True),
               random_strength=trial.suggest_float('random_strength', 0.1, 10, log=True),
               bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0))
    m = CatBoostClassifier(iterations=3000, learning_rate=0.1, eval_metric='AUC', od_type='Iter', od_wait=200, thread_count=THREADS, verbose=0,
                           cat_features=cat, random_seed=0, train_dir=str(OUT / 'catboost_info'), task_type='GPU' if GPU else 'CPU', **prm)
    m.fit(s(Xa0.iloc[sub_i]), ya0[sub_i], eval_set=(s(Xb0), yb0))
    return roc_auc_score(yb0, m.predict_proba(s(Xb0))[:, 1])
CAT_BEST = cached_search('CatBoost', cat_objective, 20).best_params
def catb_tuned(Xa, ya, Xb, yb, Xt):
    s = lambda d: d.assign(**{c: d[c].astype(str) for c in cat})
    m = CatBoostClassifier(iterations=3 * CAT_ITERS, learning_rate=0.08, eval_metric='AUC', od_type='Iter', od_wait=300, thread_count=THREADS,
                           verbose=0, cat_features=cat, random_seed=0, train_dir=str(OUT / 'catboost_info'), task_type='GPU' if GPU else 'CPU', **CAT_BEST)
    m.fit(s(Xa), ya, eval_set=(s(Xb), yb))
    return m.predict_proba(s(Xb))[:, 1], m.predict_proba(s(Xt))[:, 1]
run('CatBoost tuned', catb_tuned, 'fe')
"""),
code("""
# 2. more features, for two new members only (the feature cell above is unchanged, so no earlier member retrains)
from itertools import combinations
def plus_features(Xa, ya, Xb, Xt):
    \"\"\"Count encodings and numeric products (no labels), and in-fold target encodings of triple crosses (fitted on Xa only).\"\"\"
    out = [Xa.copy(), Xb.copy(), Xt.copy()]
    for c in ['amount_usd', 'account_balance_usd', 'hours_since_last_txn', 'distance_from_home_km', 'merchant_risk_score',
              'velocity_score', 'customer_age', 'card_age_months']:
        vc = both[c].value_counts()                                   # how common the exact value is in train + test
        for d in out:
            d['cnt_' + c] = d[c].map(vc).fillna(0).astype('float32').values
    for d in out:
        d['amount_over_balance'] = (d['amount_usd'] / (d['account_balance_usd'].abs() + 1)).astype('float32')
        d['amount_x_risk'] = (d['amount_usd'] * d['merchant_risk_score']).astype('float32')
        d['velocity_x_risk'] = (d['velocity_score'] * d['merchant_risk_score']).astype('float32')
    top6 = ['cvv_retry_count', 'ip_country_mismatch', 'is_foreign_transaction', 'is_new_merchant', 'billing_shipping_mismatch', 'auth_method']
    keys = lambda d: pd.DataFrame({f'{a}__{b}__{c}': d[a].astype(str) + '_' + d[b].astype(str) + '_' + d[c].astype(str)
                                   for a, b, c in combinations(top6, 3)})
    enc = TargetEncoder(target_type='binary', cv=5, random_state=0)
    te = [enc.fit_transform(keys(Xa), ya), enc.transform(keys(Xb)), enc.transform(keys(Xt))]
    cols = ['te3_' + c for c in keys(Xa.head(1)).columns]
    return [pd.concat([d.reset_index(drop=True), pd.DataFrame(t, columns=cols).astype('float32')], axis=1) for d, t in zip(out, te)]
def lgbm_plus(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = plus_features(Xa, ya, Xb, Xt)
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves plus', lgbm_plus, 'fe')
def xgbm_d2_plus(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = plus_features(Xa, ya, Xb, Xt)
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.05, max_depth=2, min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=300, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 2 plus', xgbm_d2_plus, 'fe')
"""),
code("""
# 3. a bigger library: k nearest neighbours (GPU) and seed bags of the two shallow XGBoost members
def knn_rate(Xa, ya, Xb, yb, Xt):
    \"\"\"Smoothed fraud rate among each row's 100 nearest training rows, on standardised numeric columns and the strongest encodings.\"\"\"
    cols = [c for c in ['merchant_risk_score', 'velocity_score', 'cvv_retry_count', 'customer_age', 'txn_count_last_24h',
                        'hours_since_last_txn', 'amount_usd', 'prior_disputes', 'te_cvv_retry_count__ip_country_mismatch',
                        'te_merchant_category', 'te_auth_method', 'te_is_new_merchant__used_vpn'] if c in Xa]
    mu, sd = Xa[cols].astype(float).mean(), Xa[cols].astype(float).std() + 1e-9
    dev = 'cuda' if GPU else 'cpu'
    ref = torch.tensor(((Xa[cols].astype(float) - mu) / sd).values, dtype=torch.float32, device=dev)
    lab = torch.tensor(ya, dtype=torch.float32, device=dev); prior = float(ya.mean())
    def rate(D):
        q = torch.tensor(((D[cols].astype(float) - mu) / sd).values, dtype=torch.float32, device=dev); res = []
        for i in range(0, len(q), 1024):
            idx = torch.cdist(q[i:i + 1024], ref).topk(100, largest=False).indices
            res.append(((lab[idx].sum(1) + 10 * prior) / 110).cpu())
        return torch.cat(res).numpy()
    return rate(Xb), rate(Xt)
if GPU:                                                   # 440k queries against 560k rows per fold: minutes on a GPU, hours on 4 CPU cores
    run('kNN rate', knn_rate, 'fe')
def xgb_bag(depth, Xa, ya, Xb, yb, Xt):
    vo, tp = 0, 0
    for seed in (1, 2, 3):
        m = xgb.XGBClassifier(n_estimators=30000, learning_rate=0.05 if depth == 2 else 0.1, max_depth=depth, min_child_weight=20, subsample=0.8,
                              colsample_bytree=0.5, random_state=seed, tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True,
                              eval_metric='auc', early_stopping_rounds=300 if depth == 2 else 500, n_jobs=THREADS)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        vo, tp = vo + m.predict_proba(Xb)[:, 1] / 3, tp + m.predict_proba(Xt)[:, 1] / 3
    return vo, tp
def xgb_d2_bag3(Xa, ya, Xb, yb, Xt): return xgb_bag(2, Xa, ya, Xb, yb, Xt)
def xgb_d1_bag3(Xa, ya, Xb, yb, Xt): return xgb_bag(1, Xa, ya, Xb, yb, Xt)
run('XGBoost depth 2 bag3', xgb_d2_bag3, 'fe')
run('XGBoost depth 1 bag3', xgb_d1_bag3, 'fe')
"""),
md("""### v10: the digits carry the label

The synthetic rows come from a generator that writes numbers digit by digit, so how a number is *written* carries signal that its size does not.
goodpjw2008 measured it ([S6E11 Part 3](https://www.kaggle.com/code/goodpjw2008/s6e11-part-3-digits-leak-the-label-lb-0-95113), LB 0.95113)
and ektarr's [RealMLP Stacking](https://www.kaggle.com/code/ektarr/lb-0-95121-realmlp-stacking) (LB 0.95121) uses the same keys.
For each of the six float columns, these keys are target-encoded inside the fold, exactly like the per-value encodings above:

| key | example for 1371.37 | what it groups |
|---|---|---|
| cents | 37 | the two decimals (also kept as a plain number) |
| int(x) | 1371 | the integer part (truncated, not rounded) |
| int(x/10) | 137 | the integer part without its last digit |
| int(10x) | 13713 | the value cut to one decimal |
| first digit, first two digits, number of digits | 1, 13, 4 | the leading digits |

Our v2-v9 features had only a part of this: the first decimal as a raw number, and *rounded* integer keys for four columns.
Rounding mixes two written forms into one key (13.6 and 14.4 both become 14), which is exactly the information the generator left; truncating keeps them apart.
Their measurement on one 4-leaf LightGBM: +0.0015 CV. The keys feed new members only, so nothing cached retrains.

The four new members, and why each:

- **LightGBM 4 leaves digits**: the best member so far (4-leaf LightGBM on the engineered set) with the digit encodings added. The direct test of the idea.
- **LightGBM 4 leaves digits s100**: the same keys with heavier smoothing (100 instead of 20), so rare keys count for less. Its errors differ slightly, which a blend can use; goodpjw2008's blend kept both.
- **XGBoost depth 2 digits**: the same inputs in another tree library; depth-2 XGBoost was the second-strongest shallow member.
- **RealMLP digits**: the neural net with the digit encodings as extra inputs. Nets make different mistakes from trees, so it is the member most likely to add to the stack.
"""),
code("""
FLOATS6 = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']
def digit_features(Xa, ya, Xb, Xt, smooth=20.0):
    \"\"\"The digit keys of the six float columns, target-encoded on Xa's labels only (5-way cross-fitted inside it); cents kept as a number.\"\"\"
    def keys(d):
        k = {}
        for c in FLOATS6:
            x = d[c].astype(float).values; i = np.floor(x + 1e-9).astype(np.int64); s = np.abs(i).astype(str)
            k[f'cents_{c}'] = np.round(x * 100).astype(np.int64) % 100
            k[f'int_{c}'], k[f'ten_{c}'], k[f'dec_{c}'] = i, np.floor(x / 10 + 1e-9).astype(np.int64), np.floor(x * 10 + 1e-6).astype(np.int64)
            k[f'lead1_{c}'] = np.array([v[:1] for v in s]); k[f'lead2_{c}'] = np.array([v[:2] for v in s]); k[f'len_{c}'] = np.array([len(v) for v in s])
        return pd.DataFrame(k).astype(str)
    ka, kb, kt = keys(Xa), keys(Xb), keys(Xt)
    enc = TargetEncoder(target_type='binary', cv=5, smooth=smooth, random_state=0)
    cols = ['dte_' + c for c in ka.columns]
    te = [enc.fit_transform(ka, ya), enc.transform(kb), enc.transform(kt)]
    cents = lambda d: pd.DataFrame({f'cents_{c}': (np.round(d[c].astype(float).values * 100) % 100).astype('float32') for c in FLOATS6})
    return [pd.concat([d.reset_index(drop=True), cents(d), pd.DataFrame(t, columns=cols).astype('float32')], axis=1)
            for d, t in zip((Xa, Xb, Xt), te)]
def lgbm_digits(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = digit_features(Xa, ya, Xb, Xt)
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves digits', lgbm_digits, 'fe')
def lgbm_digits_s100(Xa, ya, Xb, yb, Xt):                # the same with heavier smoothing: a second view the blend can use
    Xa, Xb, Xt = digit_features(Xa, ya, Xb, Xt, smooth=100.0)
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, random_state=1, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves digits s100', lgbm_digits_s100, 'fe')
def xgbm_d2_digits(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = digit_features(Xa, ya, Xb, Xt)
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=300, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 2 digits', xgbm_d2_digits, 'fe')
def realmlp_digits(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = digit_features(Xa, ya, Xb, Xt)
    return realmlp(Xa, ya, Xb, yb, Xt)
run('RealMLP digits', realmlp_digits, 'fe')
"""),
md("""### v12: three cheap members suggested by the public notebooks

Our digit members score OOF 0.9495; the public notebooks' 4-leaf members with the same keys score 0.9504-0.9505 (goodpjw2008 Part 4, megayak, bulsahkecici).
One difference we can name: every `fe` member here also carries the encodings of the 45 column pairs among the top 10.
Two public notebooks measured those pairs as useless or harmful (areebbb -0.0007 CV on 1 of 5 folds; aphoristicartist 0.95024, no gain).

- **LightGBM 4 leaves digits, no pairs**: `LightGBM 4 leaves digits` with the 45 pair encodings dropped. If the pairs only add noise, this member beats its parent; that is the whole test.
- **XGBoost depth 2 digits, no pairs**: the same test in the other tree library.
- **GLM on the encodings**: a logistic regression on the logits of every value and digit encoding. Alone it is the weakest model,
  but it fits all encodings jointly and has no trees, so its errors differ; aphoristicartist's stack gives this kind of member its largest weight (0.30), megayak's scores 0.9501 alone.
"""),
code("""
NO_PAIR = lambda d: d.drop(columns=[c for c in d.columns if c.startswith('te_') and '__' in c])
def lgbm_digits_nopair(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = digit_features(NO_PAIR(Xa), ya, NO_PAIR(Xb), NO_PAIR(Xt))
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THREADS, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves digits, no pairs', lgbm_digits_nopair, 'fe')
def xgbm_d2_digits_nopair(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = digit_features(NO_PAIR(Xa), ya, NO_PAIR(Xb), NO_PAIR(Xt))
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                          tree_method='hist', device='cuda' if GPU else 'cpu', enable_categorical=True, eval_metric='auc',
                          early_stopping_rounds=300, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 2 digits, no pairs', xgbm_d2_digits_nopair, 'fe')
def glm_encodings(Xa, ya, Xb, yb, Xt):                   # logistic regression on the logits of every value and digit encoding
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xa, Xb, Xt = digit_features(Xa, ya, Xb, Xt)
    cols = [c for c in Xa.columns if c.startswith(('te_', 'dte_'))]
    lg = lambda d: np.log(np.clip(d[cols].values, 1e-5, 1 - 1e-5) / (1 - np.clip(d[cols].values, 1e-5, 1 - 1e-5)))
    sc = StandardScaler().fit(lg(Xa))
    m = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(lg(Xa)), ya)
    return m.predict_proba(sc.transform(lg(Xb)))[:, 1], m.predict_proba(sc.transform(lg(Xt)))[:, 1]
run('GLM on the encodings', glm_encodings, 'fe')
"""),
md("""### v13: the public recipe, reproduced exactly

v12's test said the pair encodings are not why our digit members (OOF 0.9495) trail the public ones (0.9504).
Reading goodpjw2008's Part 3 and Part 5 line by line against ours shows three other differences:

| | ours (v10-v12) | goodpjw2008 Part 3 / 5 |
|---|---|---|
| encoder | sklearn `TargetEncoder`, `smooth='auto'` for the value keys | an m-estimate: (fraud count + 20 x prior) / (rows + 20), cross-fitted 5 ways inside the fold |
| what is encoded | every column, rounded keys, 45 pairs, then the digit keys | every column, the cents and every digit key, all with the same encoder |
| other inputs | first-decimal digits, original-data lookups and model, rounded keys | the raw columns and the cents only |

`smooth='auto'` picks the shrinkage from each key's variance; for a column like `account_balance_usd` (43k distinct values, a few rows each)
that is a very different amount of shrinkage than a fixed 20. The extra inputs matter too: with half the columns sampled per tree, every extra
column takes turns from the strong ones.

The three members below are that recipe, copied (credit: goodpjw2008), on our folds, starting from the raw columns:

- **LightGBM 4 leaves, Part 3 recipe**: `lgb_diglead_l4`, the public notebooks' best single model (their CV 0.95040).
- **LightGBM 3 leaves, Part 5 recipe**: the same with 3 leaves (Part 5: 3 leaves beat 4 on every fold; their 3-seed bag 0.95063).
- **XGBoost depth 2, Part 3 recipe**: `xgb_dig_d2` with the same encodings.

If the first one reaches about 0.9504 here, the encoder and the extra inputs explain our gap, and the lesson is worth more than the member.
"""),
code("""
def p3_te(Xa, ya, others, cols, smooth=20.0, inner=5, seed=0):
    \"\"\"goodpjw2008's out-of-fold m-estimate encoding: each value of each column -> (frauds + smooth x prior) / (rows + smooth).\"\"\"
    prior = ya.mean()
    Xa, others = Xa.copy(), [o.copy() for o in others]
    key = lambda X, c: X[c].astype(str).values if str(X[c].dtype) == 'category' else X[c].values
    rate = lambda k, t: (lambda s: (s['sum'] + prior * smooth) / (s['count'] + smooth))(
        pd.DataFrame({'k': k, 'y': t}).groupby('k')['y'].agg(['sum', 'count']))
    for c in cols:
        ka, enc = key(Xa, c), np.zeros(len(Xa), dtype=np.float32)
        for i, j in StratifiedKFold(inner, shuffle=True, random_state=seed).split(ka, ya):
            enc[j] = pd.Series(ka[j]).map(rate(ka[i], ya[i])).fillna(prior).values
        Xa[f'te_{c}'] = enc
        r = rate(ka, ya)
        for o in others:
            o[f'te_{c}'] = pd.Series(key(o, c)).map(r).fillna(prior).values.astype(np.float32)
    return Xa, others
def p3_frames(Xa, ya, Xb, Xt, seed=0):
    \"\"\"Part 3's digit and leading-digit keys of the six floats, every column and key encoded with p3_te, the keys then dropped.\"\"\"
    def add_keys(d):
        d = d.reset_index(drop=True).copy()
        for c in FLOATS6:
            x = d[c].astype(float)
            d[f'cents_{c}'] = (np.round(x * 100) % 100).astype('int16')
            d[f'key_int_{c}'] = np.floor(x).astype('int64'); d[f'key_ten_{c}'] = np.floor(x / 10).astype('int64')
            d[f'key_dec_{c}'] = np.floor(x * 10 + 1e-6).astype('int64')
            i = np.floor(x).astype('int64').astype(str)
            d[f'key_l1_{c}'] = (i.str.len() * 10 + i.str[0].astype(int)).values
            d[f'key_l2_{c}'] = (i.str.len() * 100 + i.str[:2].astype(int)).values
        return d
    Xa, Xb, Xt = add_keys(Xa), add_keys(Xb), add_keys(Xt)
    cols = feats + [c for c in Xa.columns if c.startswith(('cents_', 'key_'))]
    Xa, (Xb, Xt) = p3_te(Xa, ya, [Xb, Xt], cols, seed=seed)
    drop = lambda d: d[[c for c in d.columns if not c.startswith('key_')]]
    return drop(Xa), drop(Xb), drop(Xt)
P3_LGB = dict(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=1.0, max_bin=255, verbose=-1)
def lgbm_p3_l4(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = p3_frames(Xa, ya, Xb, Xt)
    m = lgb.LGBMClassifier(**P3_LGB, random_state=42, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 4 leaves, Part 3 recipe', lgbm_p3_l4, 'raw')
def lgbm_p3_l3(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = p3_frames(Xa, ya, Xb, Xt)
    m = lgb.LGBMClassifier(**{**P3_LGB, 'num_leaves': 3}, random_state=42, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('LightGBM 3 leaves, Part 5 recipe', lgbm_p3_l3, 'raw')
def lgbm_p3_l3_bag(Xa, ya, Xb, yb, Xt):
    \"\"\"The Part 5 member refitted with five encoder seeds (the inner split p3_te encodes the training rows with), averaged.\"\"\"
    vo, tp = np.zeros(len(Xb)), np.zeros(len(Xt))
    for s in range(5):
        a, b, t = p3_frames(Xa, ya, Xb, Xt, seed=s)
        m = lgb.LGBMClassifier(**{**P3_LGB, 'num_leaves': 3}, random_state=42 + s, n_jobs=THREADS)
        m.fit(a, ya, eval_set=[(b, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
        vo += m.predict_proba(b)[:, 1] / 5; tp += m.predict_proba(t)[:, 1] / 5
    return vo, tp
run('LightGBM 3 leaves, Part 5 recipe, 5 encoder seeds', lgbm_p3_l3_bag, 'raw')
def xgbm_p3_d2(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = p3_frames(Xa, ya, Xb, Xt)
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                          reg_lambda=1.0, max_bin=256, tree_method='hist', device='cuda' if GPU else 'cpu', eval_metric='auc',
                          early_stopping_rounds=300, enable_categorical=True, max_cat_to_onehot=1, random_state=42, n_jobs=THREADS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
run('XGBoost depth 2, Part 3 recipe', xgbm_p3_d2, 'raw')
"""),
md("""### v15: test predictions from one fit on all rows, a joint GLM, and trees that start from it

From v10 to v14 the stacker's held-out AUC rose 0.95019 -> 0.95091, and the board moved only 0.95095 -> 0.95101.
The OOF side is getting better; the test side is not following.
Every member so far predicts the test set as **the mean of its 5 fold models**, each trained on 80% of the rows,
with target encodings computed from those 80%.
Four public notebooks replaced that with **one more fit on all 700k rows** (encodings from all rows too, 1.25 x the folds' mean number of trees,
since there are 25% more rows), and each measured a leaderboard gain with the OOF unchanged:

| who | change | public LB |
|---|---|---|
| [goodpjw2008, Part 4](https://www.kaggle.com/code/goodpjw2008/s6e11-part-4-refit-bag-joint-glm-lb-0-95154) | same blend, refit test predictions | 0.95113 -> 0.95132 |
| [areebbb, Pooled Stack](https://www.kaggle.com/code/areebbb/s6e11-pooled-stack-fusion-lb-0-95196) | same blend, refit test predictions | 0.95135 -> 0.95142 |
| [megayak](https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack) | refit of every member, plus joint keys | 0.95104 -> 0.95145 |

Why it can help here more than usual: the median `account_balance_usd` value occurs in 3 rows, so an encoding computed from 80% of the rows
is noticeably noisier than one from 100%. The OOF cannot show this gain, by construction: only the test predictions change.
The leaderboard is the only judge, which is why it is worth a version of its own.

The three new members, all refitted this way (`run_refit` below):

- **XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit**: v13's depth-2 member, bagged over five encoder seeds like v14's LightGBM
  (which moved that member 0.95052 -> 0.95066). XGBoost runs on the GPU, so the bag costs minutes, not the 72 of the LightGBM bag.
- **Joint GLM, full refit**: one L2 logistic regression on the numbers, flags, categories and splines, **a one-hot of every exact value and every digit key
  seen at least 5 times**, and the logits of every encoding. The one-hots let the GLM correct each encoding for the others,
  and a rare value falls back to its coarser digit keys. Our v12 GLM had only the encodings.
  Design: goodpjw2008 (Part 4, `glm_joint`, CV 0.94991, two of nine blend weights) and lazyyeou (`glm_joint_96key`, 0.9501, second-largest blend weight).
- **XGBoost depth 2 boosted from the joint GLM, full refit**: the trees start from the GLM's logit (`base_margin`) instead of from the base rate,
  so they fit only what the additive model misses. Idea and measurement: megayak, v5, best single model of that notebook (0.95104 OOF), blend +0.00016 CV, LB 0.95145 -> 0.95162.
"""),
code("""
REFIT_MULT = 1.25                                        # trees in the full refit: 1.25 x the folds' mean best iteration (25% more rows)
def run_refit(name, fit_predict, mode='raw'):
    \"\"\"run() with one change: the test prediction comes from one more fit on all training rows (encodings fitted on all of them too),
    not from the mean of the 5 fold models. fit_predict(Xa, ya, Xb, yb, Xt, n_iter=None) returns (valid prediction, test prediction, best
    iteration); called with Xb=None it fits n_iter rounds on Xa without early stopping and returns only the test prediction.\"\"\"
    assert mode == 'raw'
    if (time.time() - T_START) / 3600 > BUDGET_H:
        print(f'{name}: skipped, {BUDGET_H} h budget used'); return
    fp = fingerprint(name, fit_predict, mode)
    if fp in MANIFEST and (PRED_CACHE / f'oof_{fp}.npy').exists():
        oof[name], pred[name] = np.load(PRED_CACHE / f'oof_{fp}.npy'), np.load(PRED_CACHE / f'test_{fp}.npy'); secs[name] = MANIFEST[fp]['minutes'] * 60
        _publish(fp, name, mode)
        print(f'{name}: OOF AUC {roc_auc_score(y, oof[name]):.5f}  (loaded from the dataset, trained in {MANIFEST[fp]["version"]})'); return
    t0 = time.time(); o, its = np.zeros(len(X)), []
    for k, (tr_i, va_i) in enumerate(FOLDS):
        Xa, Xb, Xt = frames(k, mode)
        vo, _, it = fit_predict(Xa, y.iloc[tr_i].values, Xb, y.iloc[va_i].values, Xt)
        o[va_i] = vo; its.append(it)
        print(f'  {name} fold {k}: AUC {roc_auc_score(y.iloc[va_i], vo):.5f}  ({it:.0f} rounds)', flush=True)
    n = int(round(np.mean(its) * REFIT_MULT))
    _, p, _ = fit_predict(X.reset_index(drop=True), y.values, None, None, XT.reset_index(drop=True), n_iter=n)
    oof[name], pred[name], secs[name] = o, p, time.time() - t0
    _publish(fp, name, mode, trained=True)
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}  ({secs[name] / 60:.1f} min; test from one fit on all {len(X):,} rows, {n} rounds)')

def xgbm_p3_d2_bag_refit(Xa, ya, Xb, yb, Xt, n_iter=None):
    \"\"\"xgbm_p3_d2 averaged over five encoder seeds (and five model seeds); with Xb None, the five fits on all rows.\"\"\"
    vo, tp, its = 0, 0, []
    for s in range(5):
        a, b, t = p3_frames(Xa, ya, Xb if Xb is not None else Xt, Xt, seed=s)
        prm = dict(learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, max_bin=256,
                   tree_method='hist', device='cuda' if GPU else 'cpu', eval_metric='auc', enable_categorical=True, max_cat_to_onehot=1,
                   random_state=42 + s, n_jobs=THREADS)
        if Xb is None:
            m = xgb.XGBClassifier(n_estimators=n_iter, **prm).fit(a, ya, verbose=False)
            tp = tp + m.predict_proba(t)[:, 1] / 5
        else:
            m = xgb.XGBClassifier(n_estimators=20000, early_stopping_rounds=300, **prm).fit(a, ya, eval_set=[(b, yb)], verbose=False)
            vo = vo + m.predict_proba(b)[:, 1] / 5; its.append(m.best_iteration + 1)
    return (None, tp, n_iter) if Xb is None else (vo, None, float(np.mean(its)))
run_refit('XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit', xgbm_p3_d2_bag_refit)

def v15_keys(d):
    \"\"\"The digit keys p3_frames encodes (cents, int, int/10, int(10x), leading one and two digits with the length), as numbers.\"\"\"
    k = {}
    for c in FLOATS6:
        x = d[c].astype(float).values; i = np.floor(x).astype(np.int64); s = pd.Series(i.astype(str))
        k[f'cents_{c}'] = np.round(x * 100).astype(np.int64) % 100
        k[f'int_{c}'], k[f'ten_{c}'], k[f'dec_{c}'] = i, np.floor(x / 10).astype(np.int64), np.floor(x * 10 + 1e-6).astype(np.int64)
        k[f'l1_{c}'] = (s.str.len() * 10 + s.str[0].astype(int)).values
        k[f'l2_{c}'] = (s.str.len() * 100 + s.str[:2].astype(int)).values
    return pd.DataFrame(k)
_JOINT = {}                                              # the boosted member reuses the GLM member's fits: same rows, same logits
def joint_glm_logits(Xa, ya, Xb, Xt, C=0.01):
    \"\"\"The joint GLM fitted on Xa: standardised (log) numbers, flags, one-hot categories, splines, a one-hot of every exact numeric value
    and every digit key seen >= 5 times in Xa, and the standardised logits of the Part 3 encodings. Returns its logit on Xa, Xb, Xt.\"\"\"
    key = (len(Xa), len(Xb), int(ya.sum()), float(Xa['amount_usd'].sum()), float(Xb['amount_usd'].sum()))
    if key in _JOINT:
        return _JOINT[key]
    a, b, t = p3_frames(Xa, ya, Xb, Xt)
    tecols = [c for c in a.columns if c.startswith('te_')]
    lg = lambda e: (lambda q: np.log(q / (1 - q)))(np.clip(e[tecols].values, 1e-6, 1 - 1e-6))
    def dense(d):
        z = d[nums].astype(float).reset_index(drop=True)
        for c in ('amount_usd', 'account_balance_usd', 'hours_since_last_txn'):
            z[c] = np.log1p(z[c].clip(lower=0))
        return z.values
    vals = lambda d: pd.concat([d[nums].astype(float).reset_index(drop=True), v15_keys(d).astype(float)], axis=1)
    sc_n, sc_e = StandardScaler().fit(dense(Xa)), StandardScaler().fit(lg(a))
    spl = SplineTransformer(n_knots=8, degree=3, knots='quantile').fit(sc_n.transform(dense(Xa)))
    oh_c = OneHotEncoder(handle_unknown='ignore').fit(Xa[cat].astype(str))
    oh_v = OneHotEncoder(handle_unknown='ignore', min_frequency=5, dtype=np.float32).fit(vals(Xa))
    def design(d, e):
        zn = sc_n.transform(dense(d))
        return sparse.hstack([sparse.csr_matrix(zn), sparse.csr_matrix(d[boo].values.astype(float)), oh_c.transform(d[cat].astype(str)),
                              sparse.csr_matrix(spl.transform(zn)), oh_v.transform(vals(d)), sparse.csr_matrix(sc_e.transform(lg(e)))]).tocsr()
    Da = design(Xa, a)
    m = LogisticRegression(C=C, max_iter=3000, tol=1e-6).fit(Da, ya)
    print(f'    joint GLM: {Da.shape[1]:,} design columns, {m.n_iter_[0]} iterations', flush=True)
    _JOINT[key] = [m.decision_function(Da), m.decision_function(design(Xb, b)), m.decision_function(design(Xt, t))]
    return _JOINT[key]
def glm_joint(Xa, ya, Xb, yb, Xt, n_iter=None):
    za, zb, zt = joint_glm_logits(Xa, ya, Xb if Xb is not None else Xt, Xt)
    return (None, zt, 0) if Xb is None else (zb, None, 0)
run_refit('Joint GLM, full refit', glm_joint)

def xgbm_glm_boost(Xa, ya, Xb, yb, Xt, n_iter=None):
    \"\"\"megayak's booster: depth-2 XGBoost on the Part 3 inputs whose starting score (base_margin) is the joint GLM's logit,
    so its trees fit only what the GLM misses. Predictions are margins (logits); AUC and the rank blend only see their order.\"\"\"
    Xv = Xb if Xb is not None else Xt
    za, zb, zt = joint_glm_logits(Xa, ya, Xv, Xt)
    a, b, t = p3_frames(Xa, ya, Xv, Xt)
    prm = dict(learning_rate=0.02, max_depth=2, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, max_bin=256,
               tree_method='hist', device='cuda' if GPU else 'cpu', eval_metric='auc', enable_categorical=True, max_cat_to_onehot=1,
               random_state=42, n_jobs=THREADS)
    if Xb is None:
        m = xgb.XGBClassifier(n_estimators=n_iter, **prm).fit(a, ya, base_margin=za, verbose=False)
        return None, m.predict(t, base_margin=zt, output_margin=True), n_iter
    m = xgb.XGBClassifier(n_estimators=20000, early_stopping_rounds=300, **prm)
    m.fit(a, ya, base_margin=za, eval_set=[(b, yb)], base_margin_eval_set=[zb], verbose=False)
    it = m.best_iteration + 1
    print(f'    fold GLM alone {roc_auc_score(yb, zb):.5f}', flush=True)
    return m.predict(b, base_margin=zb, output_margin=True, iteration_range=(0, it)), None, it
run_refit('XGBoost depth 2 boosted from the joint GLM, full refit', xgbm_glm_boost)
"""),
md("""### Public prediction banks

Three public notebooks publish their out-of-fold and test predictions, so they can join the stack as members without
retraining: **lazyyeou** (Digit-Trie TE, Joint GLM and nested stack), **nilanjansaha123** (96-key digit trie, full refit)
and **yekenot** (RealMLP via pytabkit). Credit to them for the models; the stacker below decides how much each is worth.
Measured offline before adding them: our 35 v14 members stack to CV 0.95092, and the 12 banks together add +0.00020
(the two joint GLMs +0.00012 each, the XGBoost bags +0.00011; mikhailnaumov's XGBoost added nothing and is left out).
v16 scored 0.95176 with those (v15 0.95146). v17 adds four more: **goodpjw2008** Part 6 (GLM boost, +0.00036 on top of v16),
**megayak** (four GLM-boost members, up to +0.00036), **masayakawamata** (a same-split stack with GPT-2 token keys, the largest
single gain: +0.00060) and **ektarr** (RealMLP stack, +0.00012); all four together +0.00097 on v16's 47 members, offline.
"""),
code("""BANKS = {   # member name -> (part of the source notebook's slug, OOF file, test file); test = the author's full-data refit
    'bank lazyyeou joint GLM 96 keys': ('digit-trie-te', 'glm_joint_96key_oof.npy', 'glm_joint_96key_full.npy'),
    'bank lazyyeou XGBoost d2 bag': ('digit-trie-te', 'xgb_d2_bag_oof.npy', 'xgb_d2_bag_full.npy'),
    'bank lazyyeou XGBoost l4 bag': ('digit-trie-te', 'xgb_l4_bag_oof.npy', 'xgb_l4_bag_full.npy'),
    'bank lazyyeou LightGBM l4 bag': ('digit-trie-te', 'lgb_l4_bag_oof.npy', 'lgb_l4_bag_full.npy'),
    'bank lazyyeou LightGBM s100 bag': ('digit-trie-te', 'lgb_s100_bag_oof.npy', 'lgb_s100_bag_full.npy'),
    'bank lazyyeou LightGBM raw view': ('digit-trie-te', 'lgb_rawview_oof.npy', 'lgb_rawview_full.npy'),
    'bank nilanjan joint GLM 96 keys': ('96-key', 'glm_unified_96key_oof.npy', 'glm_unified_96key_full.npy'),
    'bank nilanjan XGBoost 96 keys bag': ('96-key', 'xgb_unified_96key_bag_oof.npy', 'xgb_unified_96key_bag_full.npy'),
    'bank nilanjan LightGBM 96 keys bag': ('96-key', 'lgb_unified_96key_bag_oof.npy', 'lgb_unified_96key_bag_full.npy'),
    'bank nilanjan LightGBM s100 bag': ('96-key', 'lgb_unified_s100_bag_oof.npy', 'lgb_unified_s100_bag_full.npy'),
    'bank nilanjan LightGBM l7': ('96-key', 'lgb_unified_l7_oof.npy', 'lgb_unified_l7_full.npy'),
    'bank goodpjw2008 GLM boost (Part 6)': ('s6e11-part-', 'oof_gb_s0.npy', 'refit_gb_s0.npy'),          # matched on the series prefix: the author renames the kernel with every part (6, 7, 8); same files
    'bank goodpjw2008 joint GLM (Part 6)': ('s6e11-part-', 'glmoof_gb_s0.npy', 'glmrefit_gb_s0.npy'),
    'bank megayak GLM boost': ('per-value-digit-te', 'oof_glm_boost.npy', 'refit_glm_boost.npy'),
    'bank megayak GLM boost XGBoost': ('per-value-digit-te', 'oof_glm_boost_xgb.npy', 'refit_glm_boost_xgb.npy'),
    'bank megayak GLM boost raw': ('per-value-digit-te', 'oof_glm_boost_raw.npy', 'refit_glm_boost_raw.npy'),
    'bank megayak GLM': ('per-value-digit-te', 'oof_glm_gen.npy', 'refit_glm_gen.npy'),
    'bank masayakawamata stack (GPT-2 keys)': ('logreg-stacking-same-split', 'oof.npy', 'test.npy'),
}
CSV_BANKS = {   # member name -> (slug part, OOF csv, its column, test csv); rows sorted by id
    'bank yekenot RealMLP': ('realmlp-pytabkit', 'oof_preds.csv', 'is_fraud', 'submission.csv'),
    'bank ektarr RealMLP stack': ('realmlp-stacking', 'oof_predictions.csv', 'stack', 'submission.csv'),
}
import glob as _glob
def _find(name, owner):
    hits = [h for h in _glob.glob(f'/kaggle/input/**/{name}', recursive=True) if owner in h.lower()]
    return min(hits, key=len) if hits else None
for name, (owner, fo, ft) in BANKS.items():
    po, pt = _find(fo, owner), _find(ft, owner)
    if po is None or pt is None:
        print(f'{name}: not mounted, skipped'); continue
    o, t = np.load(po).ravel(), np.load(pt).ravel()
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    oof[name], pred[name], secs[name] = o, t, 0.0          # trained elsewhere: no minutes here
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}')
for name, (owner, fo, col, ft) in CSV_BANKS.items():
    po, pt = _find(fo, owner), _find(ft, owner)
    if po is None or pt is None:
        print(f'{name}: not mounted, skipped'); continue
    o = pd.read_csv(po).sort_values('id')[col].values; t = pd.read_csv(pt).sort_values('id').iloc[:, -1].values
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    oof[name], pred[name], secs[name] = o, t, 0.0
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}')
"""),
md("""### v18: a GLM booster trained on ten folds

**What it is.** **sameershaikh27**'s [GLM Boost on Ten Folds](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-glm-boost-on-ten-folds)
(OOF 0.95159, LB 0.95118 alone): a plain GLM on the encoding logits, then 4-leaf LightGBM trees that start from the GLM's logit,
bagged over two encoder seeds. Its test prediction is a refit on all rows. Credit to Sameer Shaikh for the model and the published predictions.

**What is different.** Every other GLM booster in this stack trains on five folds, so each fold's encodings come from 80% of the rows.
This one uses ten folds: each fold's encodings come from 90% of the rows. Rare values (the median `account_balance_usd` value has 3 rows)
get a rate from more rows, so this member makes different errors on the rare rows.

**Why it is in.** We measured it offline before adding it: v17's 58 members, the same logistic stacker, the same 5-fold split.
It adds **+0.00015** (0.95210 -> 0.95224), the only one of 11 new public banks to pass our bar of +0.00005.
Its highest rank correlation with any member already in the stack is 0.9922. The 5-fold GLM boosters (goodpjw2008's 5-seed bag, 0.9989) add only +0.00004.
Same model, same data, a different number of folds: the encoder's training share is a free knob that adds diversity to a stack.

**What the board said (v21).** It did not hold up. With it, the stacker's held-out AUC rose and the public LB fell, twice:

| version | members | stacker held-out | public LB |
|---|---|---|---|
| v17 | 58, without the ten-fold member | 0.95212 | **0.95211** |
| v18 | 57 + the ten-fold member (two banks not mounted) | 0.95225 | 0.95195 |
| v20 | 58 + the ten-fold member | 0.95226 | 0.95202 |

goodpjw2008 measured the reason in [Part 9.2](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298):
a member's out-of-fold predictions come from models that saw 90% of the rows, but its test predictions are a refit on 100% of the rows, the same as every 5-fold member's.
So ten folds improve only the validation side. The stacker reads the 90%-of-rows OOF, believes the member is stronger than it will be on the test rows,
and gives it the second-largest weight (0.416 in v20). Credit to goodpjw2008 for the measurement, and to Sadam Torres, who first noticed the large weight.
The lesson for any stack: **every member's OOF must be made under the same conditions**, or the stacker's weights reward the difference in conditions, not the model.
The public 20% has a noise of about ±0.00035 (goodpjw2008 Part 7.4), so two scores are evidence, not proof; the member stays in v21 so that v21 changes one thing (the CNN).

**A correction (v22).** goodpjw2008 withdrew the strong form of this in Part 11.3: their wide stack contains ten-fold GLM boosts, its CV rose and its public score rose with it,
and "two public scores 0.00004 and 0.00007 apart cannot show an illusion". The test-side argument above still holds; the verdict does not.
In our v22 wide pool the ten-fold member takes the second-largest weight again (0.291 offline).
"""),
code("""V18_BANKS = {   # member name -> (part of the source notebook's slug, OOF file, test file); test = the author's full-data refit
    'bank sameershaikh27 GLM boost, ten folds': ('ten-folds', 'gb10_bag_oof.npy', 'gb10_bag_full.npy'),
}
for name, (owner, fo, ft) in V18_BANKS.items():
    po, pt = _find(fo, owner), _find(ft, owner)
    if po is None or pt is None:
        print(f'{name}: not mounted, skipped'); continue
    o, t = np.load(po).ravel(), np.load(pt).ravel()
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    oof[name], pred[name], secs[name] = o, t, 0.0
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}')
"""),
md("""### v21: a CNN that reads the digits

**What it is.** **queyrusi**'s [shared digit CNN](https://www.kaggle.com/code/queyrusi/fraud-a-shared-digit-cnn)
(OOF 0.94974, public LB 0.95132 in their 90/10 blend with trees). One small convolutional net reads the decimal string of each of 13 numeric columns,
character by character, with the same filters for every column. A DeepSets layer adds the per-column scores, and embeddings carry the categoricals.
Three seeds, five folds on the community split `StratifiedKFold(5, shuffle=True, random_state=42)`; the test side is the mean rank of all 15 models.
The predictions come from the author's dataset `queyrusi/fraud-shared-digit-cnn-weights`. Credit to queyrusi for the model, the weights and the cached predictions.

**What it does differently.** Every other member reads the digits through target encodings: a key such as `int(x/10)` is replaced by the fraud rate of rows with that key.
The CNN never sees a fraud rate. It learns its own digit patterns from the strings, so it can make errors the encoded members do not make.

**Why it is in.** We measured it offline before adding it, on v20's 59 members with the same stacker:
stacker held-out **+0.00010** on one 5-fold split of the OOF matrix and **+0.00005** on a second (0.95223 -> 0.95233 and 0.95226 -> 0.95231), at our bar of +0.00005 on both.
Its highest rank correlation with any member in the stack is 0.9795; the encoded tree banks sit at 0.99+ with each other.
goodpjw2008 ([Part 9.4](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298)) measured the same model on their stack: nested CV +0.00007, public LB 0.95235 -> 0.95224.
Their public 20% has a noise of about ±0.00035, so that one score does not decide it; v21 measures it on our stack.
"""),
code("""V21_NPZ_BANKS = {   # member name -> (part of the dataset path, npz file); the npz holds ids_train, ids_test, oof, test (ranks)
    'bank queyrusi digit CNN bag': ('digit-cnn', 'predictions.npz'),
}
for name, (owner, fz) in V21_NPZ_BANKS.items():
    pz = _find(fz, owner)
    if pz is None:
        print(f'{name}: not mounted, skipped'); continue
    z = np.load(pz)
    o = pd.Series(z['oof'], index=z['ids_train']).loc[train['id']].values      # align on id, not on row order
    t = pd.Series(z['test'], index=z['ids_test']).loc[test['id']].values
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    oof[name], pred[name], secs[name] = o, t, 0.0
    print(f'{name}: OOF AUC {roc_auc_score(y, o):.5f}')
"""),
md("""### v22: when a source notebook's output disappears

A bank member is read from the newest version of its author's notebook. On 2026-11-06 masayakawamata reran
[LogReg Stacking, Same-Split OOFs](https://www.kaggle.com/code/masayakawamata/s6e11-logreg-stacking-same-split-oofs) as a code-only update, and that version has no output files.
Our stack's largest weight (0.736 in v21) would have dropped out without an error: the loader above prints "not mounted, skipped" and goes on.
The author kept the old files in the dataset [S6E11 Public Stack v10 Output Archive](https://www.kaggle.com/datasets/masayakawamata/s6e11-public-stack-v10-outputs).
We checked them against the copy v17-v21 used: `oof.npy` and `test.npy` are byte-identical (OOF AUC 0.95153). Credit and thanks to Masaya Kawamata for keeping the archive.
The cell below reads the archive only if the notebook output is missing. The lesson for anyone stacking public banks: **print every member you expected and did not get**, and count them before the stacker runs.
"""),
code("""V22_ARCHIVE_BANKS = {   # member name -> (part of the dataset path, OOF file, test file); used only if the notebook output did not mount
    'bank masayakawamata stack (GPT-2 keys)': ('public-stack-v10-outputs', 'oof.npy', 'test.npy'),
}
for name, (owner, fo, ft) in V22_ARCHIVE_BANKS.items():
    if name in oof:
        print(f'{name}: from the notebook output'); continue
    po, pt = _find(fo, owner), _find(ft, owner)
    if po is None or pt is None:
        print(f'{name}: archive not mounted either, skipped'); continue
    o, t = np.load(po).ravel(), np.load(pt).ravel()
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    oof[name], pred[name], secs[name] = o, t, 0.0
    print(f'{name}: from the archive, OOF AUC {roc_auc_score(y, o):.5f}')
expected = list(BANKS) + list(CSV_BANKS) + list(V18_BANKS) + list(V21_NPZ_BANKS)
print('bank members expected', len(expected), '| loaded', sum(k in oof for k in expected), '| missing', [k for k in expected if k not in oof])
"""),
md("""### v22: frozen copies of a bank that keeps changing

**What happened.** megayak reran [Per-Value + Digit TE | Shallow-Tree Stack](https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack)
on 2026-11-05 and again on 2026-11-07. The 11-05 version no longer writes `glm_gen` (our member "bank megayak GLM"), and its `glm_boost_raw` changed
(OOF AUC 0.95100 -> 0.95115, rank correlation with the old file 0.99645). The loader above reads the newest version, so a push would have changed our stack silently.

**What we do.** The four files v17-v21 stacked are frozen in our dataset [S6E11 frozen public prediction banks](https://www.kaggle.com/datasets/alexchilton/s6e11-frozen-public-banks),
byte for byte as megayak published them on 2026-11-04 (OOF AUC 0.95137, 0.95132, 0.95100, 0.95047, the same numbers v21's log printed).
The cell below reads them in place of the live notebook. The live notebook's new members (`glm_cnt`, `glm_boost_c`) go into the wide pool further down.
Credit for the models stays with megayak; please upvote the source.
"""),
code("""V22_FROZEN_BANKS = {   # member name -> (OOF file, test file) in our frozen dataset; replaces whatever the live source publishes today
    'bank megayak GLM boost': ('oof_glm_boost.npy', 'refit_glm_boost.npy'),
    'bank megayak GLM boost XGBoost': ('oof_glm_boost_xgb.npy', 'refit_glm_boost_xgb.npy'),
    'bank megayak GLM boost raw': ('oof_glm_boost_raw.npy', 'refit_glm_boost_raw.npy'),
    'bank megayak GLM': ('oof_glm_gen.npy', 'refit_glm_gen.npy'),
}
for name, (fo, ft) in V22_FROZEN_BANKS.items():
    po, pt = _find(fo, 'frozen-public-banks'), _find(ft, 'frozen-public-banks')
    if po is None or pt is None:
        print(f'{name}: frozen copy not mounted, kept the live one' if name in oof else f'{name}: not mounted, skipped'); continue
    o, t = np.load(po).ravel(), np.load(pt).ravel()
    assert len(o) == len(X) and len(t) == len(XT), (name, len(o), len(t))
    live = 'live file differs' if name in oof and not np.array_equal(oof[name], o) else 'same as live' if name in oof else 'gone from the live source'
    oof[name], pred[name], secs[name] = o, t, 0.0
    print(f'{name}: frozen copy, OOF AUC {roc_auc_score(y, o):.5f} ({live})')
print('bank members expected', len(expected), '| loaded', sum(k in oof for k in expected), '| missing', [k for k in expected if k not in oof])
"""),
md("""### Blend

Each model's out-of-fold predictions are turned into ranks (AUC only sees order). Two ways to combine them, compared on rows they did not fit:

- a **hill climb** that adds, with replacement, whichever model raises the blend's OOF AUC most;
- a **logistic-regression stacker** on the members' logit-ranks, the combiner every s6e8 and s6e9 top write-up used.

Both are refitted inside a 5-fold split of the OOF matrix, so neither is scored on the rows it was fitted on; the better one makes the submission.
"""),
code("""
from scipy.stats import rankdata
R = {k: rankdata(v) / len(v) for k, v in oof.items()}; RT = {k: rankdata(v) / len(v) for k, v in pred.items()}
names = list(R)                                           # the hill climb gives a weaker LightGBM variant weight 0 itself
display(pd.DataFrame({k: [roc_auc_score(y, oof[k]), secs[k] / 60] for k in R}, index=['OOF AUC', 'minutes']).T.round(5))
display(pd.DataFrame(np.corrcoef([R[k] for k in names]), index=names, columns=names).round(3))
w = {k: 0 for k in names}; best, cur = -1, np.zeros(len(X))
for step in range(30):
    cand = {k: roc_auc_score(y, (cur * step + R[k]) / (step + 1)) for k in names}
    k = max(cand, key=cand.get)
    if cand[k] <= best + 1e-7: break
    best = cand[k]; w[k] += 1; cur = (cur * step + R[k]) / (step + 1)
w = {k: v / sum(w.values()) for k, v in w.items()}
print('weights', {k: round(v, 3) for k, v in w.items()}, f'| blend OOF AUC {best:.5f} | equal-weight rank mean {roc_auc_score(y, np.mean([R[k] for k in names], 0)):.5f}')
hc_test = sum(w[k] * RT[k] for k in names)

# a logistic-regression stacker on the members' logit-ranks, scored by its own 5-fold CV over the OOF matrix (so it is not judged on rows it fitted)
from scipy.special import logit
Z = np.column_stack([logit(np.clip(R[k], 1e-6, 1 - 1e-6)) for k in names]); ZT = np.column_stack([logit(np.clip(RT[k], 1e-6, 1 - 1e-6)) for k in names])
st_runs = []                                              # three outer splits: the spread is the noise of this comparison
st_hold = {}                                              # v25: the held-out predictions per split, for the two-stack mix below
for seed in (7, 8, 9):
    st_oof = np.zeros(len(Z))
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y):
        st_oof[va_i] = LogisticRegression(C=0.05, max_iter=1000).fit(Z[tr_i], y.iloc[tr_i]).decision_function(Z[va_i])
    st_runs.append(roc_auc_score(y, st_oof)); st_hold[seed] = st_oof.copy()
st_auc = st_runs[0]                                       # split 7, the one the hill climb is refitted on below
print(f'logistic stacker held-out AUC over 3 outer splits: {np.mean(st_runs):.5f} +- {np.std(st_runs):.5f}', np.round(st_runs, 5))
# the hill climb's own number is fitted on the same OOF it is scored on; re-fit it per fold the same way for a fair comparison
hc_oof = np.zeros(len(Z))
for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y):
    ww = {k: 0 for k in names}; bb, cc = -1, np.zeros(len(tr_i))
    for step in range(30):
        cand = {k: roc_auc_score(y.iloc[tr_i], (cc * step + R[k][tr_i]) / (step + 1)) for k in names}
        k = max(cand, key=cand.get)
        if cand[k] <= bb + 1e-7: break
        bb = cand[k]; ww[k] += 1; cc = (cc * step + R[k][tr_i]) / (step + 1)
    hc_oof[va_i] = sum(ww[k] * R[k][va_i] for k in names) / max(1, sum(ww.values()))   # normalised: folds must share one scale
hc_auc = roc_auc_score(y, hc_oof)
stk = LogisticRegression(C=0.05, max_iter=1000).fit(Z, y)
print(f'held-out AUC: hill climb {hc_auc:.5f}, logistic stacker {st_auc:.5f}; stacker weights', dict(zip(names, np.round(stk.coef_[0], 3))))
blend = stk.decision_function(ZT) if st_auc > hc_auc else hc_test
print('submitting the', 'logistic stacker' if st_auc > hc_auc else 'hill climb')
sub = pd.DataFrame({'id': test['id'], TARGET: rankdata(blend) / len(blend)}); sub.to_csv(OUT / 'submission.csv', index=False)
print(sub.shape, sub.head(3).to_dict('records'))
"""),
md("""### What each member is worth to the blend

Left: each model alone. Right: how much the stacker's held-out AUC falls when that one model is removed and the stacker is refitted.
A model can be strong on the left and worth nothing on the right, when another member already carries the same information.
"""),
code("""
drop = {}                                                 # 3 folds here (one refit per member), so the base is refitted the same way
base3 = np.zeros(len(Z))
for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=7).split(Z, y):
    base3[va_i] = LogisticRegression(C=0.05, max_iter=1000).fit(Z[tr_i], y.iloc[tr_i]).decision_function(Z[va_i])
base3 = roc_auc_score(y, base3)
for j, k in enumerate(names):
    keep = [i for i in range(len(names)) if i != j]; o = np.zeros(len(Z))
    for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=7).split(Z, y):
        o[va_i] = LogisticRegression(C=0.05, max_iter=1000).fit(Z[tr_i][:, keep], y.iloc[tr_i]).decision_function(Z[va_i][:, keep])
    drop[k] = base3 - roc_auc_score(y, o)
own = pd.Series({k: roc_auc_score(y, oof[k]) for k in names}).sort_values()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.38 * len(names) + 1.5), sharey=True)
a1.barh(own.index, own.values, color='#9bb7d4'); a1.set_xlim(own.min() - 0.002, own.max() + 0.0005); a1.set_title('OOF AUC alone')
d = pd.Series(drop)[own.index]; a2.barh(d.index, d.values * 1e4, color=['#4a6fa5' if v > 0 else '#c0392b' for v in d.values])
a2.axvline(0, color='#999', lw=0.8); a2.set_title('AUC lost when removed (x 1e-4)'); plt.tight_layout(); plt.show()
"""),
md("""### v22: a wide pool of public members, and why the top public notebooks score above our stack

**The question.** On 2026-11-07 the best public notebooks scored 0.95277-0.95285 and our best was 0.95211.
[goodpjw2008 Part 11](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298) (LB 0.95285; the notebook is now called Part 12) is a 70 / 30 rank blend of two stacks;
[sameershaikh27's Wide Pooled Stack](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-wide-pooled-stack-every-bank-found) (nested CV 0.95241, LB 0.95277) is one stack of 178 public members.
Is the gain in the blend of public files, or in the width of the pool? We measured both on our own OOF.

**What it is.** Every public bank published on the same folds (`StratifiedKFold(5, shuffle=True, random_state=42)`) that this stack did not have yet:
sadamtorres' [TabPFN and GBDT pairs](https://www.kaggle.com/datasets/sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs) (14),
masayakawamata's [GPT-2 target-first classifier](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof) and [TabPack](https://www.kaggle.com/datasets/masayakawamata/s6e11-tabpack-cpu-starter-predictions),
sameershaikh27's [digit-sequence CNN](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-digit-sequence-cnn-member),
wguesdon's [tree ensemble predictions](https://www.kaggle.com/datasets/wguesdon/s6e11-tree-ensemble-predictions) (8),
malyshevdanil's [pooled stack v9 bank](https://www.kaggle.com/datasets/malyshevdanil/s6e11-pooled-stack-v9) (32),
goodpjw2008's [OOF bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank) (70) and megayak's two new GLM members.
Where a bank publishes its fold ids, the cell checks that they are our partition.
A member enters if its OOF AUC is within 0.008 of the best and its rank correlation with every member already in is below 0.99995 (sameershaikh27's and goodpjw2008's rules).
The same logistic stacker as above then runs on all of them, at `C = 0.01` instead of 0.05.

**What we measured offline** (v21's 60 members, held-out AUC of the stacker on outer splits 7 and 11; `scratch/banks/wide_1107.py` in github.com/alexchilton/kaggley):

| arm | split 7 | split 11 |
|---|---|---|
| v21's 60 members, C 0.05 (v21 as submitted) | 0.95233 | 0.95232 |
| v21's 60 members, C 0.01 | 0.95231 | 0.95230 |
| 60 + 106 public members, C 0.05 | 0.95246 | 0.95238 |
| **60 + 106 public members, C 0.01** | **0.95250** | **0.95245** |
| 60 + goodpjw2008's wide stack as one member, C 0.05 | 0.95241 | 0.95240 |
| goodpjw2008's wide stack alone (its nested OOF) | 0.95250 | 0.95250 |
| rank blend 0.7 goodpjw2008 wide stack + 0.3 our wide stack | 0.95252 | 0.95251 |

**What it says.**
1. **The width is the gain; the public-file blend is not.** 106 more members add +0.00017 / +0.00013. Blending our wide stack with goodpjw2008's adds +0.00002 / +0.00006 on the OOF.
   goodpjw2008's blend moved the public board +0.00018 (0.95267 -> 0.95285), inside the ±0.0003 that a 20% public sample moves by chance.
   Our wide stack alone reaches what goodpjw2008's reaches alone (0.95250).
2. **One member at a time understates a pool.** We tested the GPT-2 target-first member alone on 2026-11-03: -0.00010. In the wide pool it takes the largest stacker weight (0.461 offline), then
   sameershaikh27's ten-fold GLM boost (0.291) and sadamtorres' GLM-boost pairs (0.235, 0.216). HYPOTHESIS (unverified): these members correct each other,
   and a member whose errors only help once another member is there cannot pass a one-at-a-time test.
3. **Regularisation has to grow with width.** `C = 0.01` costs 0.00002 on 60 members and gains 0.00004-0.00007 on 166.

**Credit.** Sameer Shaikh built the first wide pool and the gate; goodpjw2008 rebuilt it and published the blend; every bank author above trained the members. Please upvote them.
The cell below submits the wide stack if its held-out AUC over splits 7, 8 and 9 is above the 60-member stacker's.
"""),
code("""import os, json, glob as _g
from joblib import Parallel, delayed
pool = {}                                                 # member name -> (OOF in train order, test in test order)
def _pool_add(n, o, t):
    o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
    if len(o) == len(X) and len(t) == len(XT) and np.isfinite(o).all() and np.isfinite(t).all(): pool[n] = (o, t)
    else: print('  skipped', n, len(o), len(t))
def _dir(fname, owner):
    p = _find(fname, owner); return None if p is None else os.path.dirname(p)
fold42 = np.zeros(len(X), int)
for f_, (_, va_i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(X)), y)): fold42[va_i] = f_
def _same_folds(f): return all(len(set(f[fold42 == k])) == 1 for k in range(5))
tr_id, te_id = train['id'].values, test['id'].values
def _by_id(df, ids): return df.set_index('id').loc[ids].iloc[:, -1].values
d = _dir('pair_oof_v5_glm.csv', 'tabpfn-gbdt-oof-test-pairs')
for p in sorted(_g.glob(f'{d}/pair_oof_*.csv')) if d else []:
    n = os.path.basename(p)[len('pair_oof_'):-4]
    _pool_add(f'sadamtorres {n}', _by_id(pd.read_csv(p), tr_id), _by_id(pd.read_csv(f'{d}/pair_test_{n}.csv'), te_id))
d = _dir('fold_ids.npy', 'gpt2-target-first-oof')
if d and _same_folds(np.load(f'{d}/fold_ids.npy')): _pool_add('masayakawamata GPT-2 target-first', np.load(f'{d}/oof.npy'), np.load(f'{d}/test.npy'))
d = _dir('oof_tabpack.csv', 'tabpack-cpu-starter')
if d:
    to = pd.read_csv(f'{d}/oof_tabpack.csv').set_index('id').loc[tr_id]
    if _same_folds(to['fold'].values): _pool_add('masayakawamata TabPack', to['is_fraud'].values, pd.read_csv(f'{d}/test_tabpack.csv').set_index('id').loc[te_id, 'is_fraud'].values)
d = _dir('dcnn_bag_oof.npy', 'digit-sequence-cnn')
if d: _pool_add('sameershaikh27 digit-sequence CNN', np.load(f'{d}/dcnn_bag_oof.npy'), np.load(f'{d}/dcnn_bag_full.npy'))
d = _dir('folds.csv', 'tree-ensemble-predictions')
if d and _same_folds(pd.read_csv(f'{d}/folds.csv').set_index('id').loc[tr_id, 'fold'].values):
    for p in sorted(_g.glob(f'{d}/*.npz')):
        z = np.load(p); n = os.path.basename(p)[:-4]
        o = z['stack_oof'] if 'stack_oof' in z else z['seed_oof'].mean(1) if 'seed_oof' in z else z['oof']
        t = z['stack_test'] if 'stack_test' in z else z['fold_test'].mean(0)
        _pool_add(f'wguesdon {n}', pd.Series(o, index=z['train_id']).loc[tr_id].values, pd.Series(t, index=z['test_id']).loc[te_id].values)
d = _dir('pool_bank.npz', 'pooled-stack-v9')
if d:
    z = np.load(f'{d}/pool_bank.npz'); mn = json.load(open(f'{d}/pool_manifest.json'))['names']
    if (z['y'] == y.values).all():
        for j, n in enumerate(mn): _pool_add(f'malyshevdanil {n}', z['oof'][:, j], z['test'][:, j])
d = _dir('folds.csv', 's6e11-oof-bank')
if d and _same_folds(pd.read_csv(f'{d}/folds.csv').set_index('id').loc[tr_id, 'fold'].values):
    for p in sorted(_g.glob(f'{d}/oof_*.npy')):
        n = os.path.basename(p)[4:-4]
        if os.path.exists(f'{d}/test_{n}.npy'): _pool_add(f'goodpjw2008 {n}', np.load(p), np.load(f'{d}/test_{n}.npy'))
for n in ['glm_cnt', 'glm_boost_c']:
    po, pt = _find(f'oof_{n}.npy', 'per-value-digit-te'), _find(f'refit_{n}.npy', 'per-value-digit-te')
    if po and pt: _pool_add(f'megayak {n}', np.load(po), np.load(pt))
po, pt = _find('lm_oof.npy', 'target-first-lm'), _find('lm_test.npy', 'target-first-lm')   # v23: megayak's target-first language model
if po and pt: _pool_add('megayak target-first LM', np.load(po), np.load(pt))
else: print('megayak target-first LM: not mounted, skipped')
src = pd.Series([k.split()[0] for k in pool]).value_counts()
print(len(pool), 'public members loaded:', src.to_dict())

def _std(v): r = rankdata(v); return ((r - r.mean()) / r.std()).astype(np.float32)
pauc = {k: roc_auc_score(y, v[0]) for k, v in pool.items()}
top = max(max(pauc.values()), max(roc_auc_score(y, oof[k]) for k in names))
S = np.stack([_std(oof[k]) for k in names]); kept = []
for k in sorted(pool, key=pauc.get, reverse=True):
    if pauc[k] < top - 0.008: continue
    s = _std(pool[k][0])
    if (S @ s / len(s)).max() < 0.99995: kept.append(k); S = np.vstack([S, s])
del S
print(f'best single member {top:.5f}; public members within 0.008 and not near-copies: {len(kept)}')
lr_ = lambda v: logit(np.clip(rankdata(v) / len(v), 1e-6, 1 - 1e-6))
ZW = np.column_stack([Z] + [lr_(pool[k][0]) for k in kept]); ZWT = np.column_stack([ZT] + [lr_(pool[k][1]) for k in kept])
wnames = names + kept
def _fold(A, yy, tr_i, va_i): return va_i, LogisticRegression(C=0.01, max_iter=3000, tol=1e-6).fit(A[tr_i], yy[tr_i]).decision_function(A[va_i])
wide_runs = []
wide_hold = {}                                            # v25: kept for the two-stack mix below
for seed in (7, 8, 9):                                    # the same three outer splits as the 60-member stacker above
    s = np.zeros(len(ZW))
    for va_i, dv in Parallel(n_jobs=-1)(delayed(_fold)(ZW, y.values, tr_i, va_i) for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(ZW, y)): s[va_i] = dv
    wide_runs.append(roc_auc_score(y, s)); wide_hold[seed] = s
print(f'held-out AUC over splits 7, 8, 9: {len(names)}-member stacker {np.mean(st_runs):.5f} {np.round(st_runs, 5)} | wide stacker ({len(wnames)} members, C 0.01) {np.mean(wide_runs):.5f} {np.round(wide_runs, 5)}')
wstk = LogisticRegression(C=0.01, max_iter=3000, tol=1e-6).fit(ZW, y)
wcoef = pd.Series(wstk.coef_[0], index=wnames)
print('largest wide-stacker weights:'); print(wcoef.reindex(wcoef.abs().sort_values(ascending=False).index).head(15).round(3).to_string())
def _group(k):
    if k == 'megayak target-first LM': return 'megayak LM (new in v23)'
    if k in pool: return k.split()[0] + ' (new in v22)'
    return 'public banks of v16-v21' if k.startswith('bank ') else 'trained in this notebook'
g = wcoef.abs().groupby([_group(k) for k in wnames]).sum().sort_values()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
a1.barh(g.index, g.values, color=['#c98b2b' if 'new' in i else '#4a6fa5' for i in g.index]); a1.set_title('sum of |stacker weight| by source, wide stack')
a2.bar([f'{len(names)} members\\nC 0.05', f'{len(wnames)} members\\nC 0.01'], [np.mean(st_runs), np.mean(wide_runs)], color=['#4a6fa5', '#c98b2b'],
       yerr=[np.std(st_runs), np.std(wide_runs)], capsize=6)
a2.set_ylim(min(np.mean(st_runs), np.mean(wide_runs)) - 0.0003, max(np.mean(st_runs), np.mean(wide_runs)) + 0.0002); a2.set_title('stacker held-out AUC, splits 7 / 8 / 9')
plt.tight_layout(); plt.show()
if np.mean(wide_runs) > np.mean(st_runs):
    blend = wstk.decision_function(ZWT)
    sub = pd.DataFrame({'id': test['id'], TARGET: rankdata(blend) / len(blend)}); sub.to_csv(OUT / 'submission.csv', index=False)
    print('submitting the wide stacker', sub.shape)
else:
    print('the wide stacker did not beat the 60-member stacker here; submission.csv is the one written above')
"""),
md("""### v23: a second language model in the wide pool (megayak's target-first GPT-2)

**What it is.** [megayak](https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack) trained their own GPT-2 small on one laptop GPU
and published its out-of-fold and test predictions: [S6E11 target-first LM predictions](https://www.kaggle.com/datasets/megayak/s6e11-target-first-lm) (CC0), on our five folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).
v23 adds it to the wide pool as one more member. Nothing else changes.

**How it works.** Each row is written as text, **label first**: `is_fraud` comes before the columns, then one learned token per column name followed by the GPT-2 tokens of its value, in a fresh random column order each time.
The model learns to write fraud rows and legitimate rows. To score a row, it reads the row twice, once after "fraud" and once after "legit", and compares the two likelihoods (Bayes' rule turns them into log-odds).
The idea is Chris Deotte's (S6E10 1st place); masayakawamata's public S6E11 member uses the same recipe with different settings, and megayak's row format is more compact.

**Why it should help.** A language model reads the digits of every number in the order the generator wrote them; trees and GLMs see encodings of values and digit keys.
Measured here (`scratch/banks/wide_1108.py` in github.com/alexchilton/kaggley):
- alone it scores OOF **0.95043**, below our best members (0.95159);
- its rank correlation with masayakawamata's GPT-2 member is **0.9802** (OOF), and with every member already in our stack at most **0.9824**. Our tree members correlate with each other at 0.99+.
A member that is weaker alone but ranks rows differently is what a stack can use (lesson 1 below; v22's GPT-2 member took the largest weight).

**Who found it.** [goodpjw2008 Part 12](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298) added it to their 200-member wide stack:
nested CV 0.952504 -> 0.952565, public LB 0.95285 -> **0.95298**, 4 of 5 folds up. They found it because
[areebbb](https://www.kaggle.com/code/areebbb/s6e11-wide-stack-shallow-tree-fusion-0-95291) had reached 0.95291 by mixing megayak's *submission file* into goodpjw2008's files at 20%, with the weight picked on the public board.
megayak's file is far from the others because of this language model. Mixing a file in can only be judged on the public board; out-of-fold predictions can go into the stack and be measured on all 700k rows.
Credit: megayak for training and publishing the model, goodpjw2008 for the out-of-fold test, areebbb for the blend that pointed at it, Chris Deotte and masayakawamata for the recipe. Please upvote them.

**What to read in the output.** The wide-stacker line of the cell above now includes this member. v22 printed **0.95251** for the same three splits (7 / 8 / 9) without it,
and every other source in the pool was checked byte-identical to v22's copy on 2026-11-08. The difference between the two lines is what this member adds; the weights line shows its stacker weight.
"""),
md("""### v25: the 0.95300 public recipe, measured on our own two stacks

**The question.** On 2026-11-08 three public notebooks reached public LB **0.95300** with the same two files:
[malyshevdanil's Fraud Signal Lab](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) (v5, the recipe),
[areebbb's Wide Stack + Shallow-Tree Fusion](https://www.kaggle.com/code/areebbb/s6e11-wide-stack-shallow-tree-fusion-0-95300) and
[lucifer19's Golden Signal V22](https://www.kaggle.com/code/lucifer19/the-golden-signal-consensus-blend-v22).
Our best is 0.95260. What is in that recipe, and does it hold on rows a model did not fit?

**What it is.** 65% of [goodpjw2008 Part 12](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298)'s wide stack (201 public and own members, one logistic stacker)
and 35% of goodpjw2008's **Part 9**, their earlier, narrower stack of their own members, with the test side averaged over every refit.
Both files are turned into ranks, then logits (`logit((rank - 0.5) / n)`), and mixed. Nothing is trained.
The files are frozen in [s6e11-fraud-series-artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts); the cell below checks their SHA-256 against the hashes the 0.95300 notebooks check.

**What goodpjw2008 measured themselves** (Part 11, section 11.2, the same mix with the older wide stack):

| weight of the wide stack | nested CV | public LB |
|---|---|---|
| 1 | 0.952504 | 0.95267 |
| 0.8 | 0.952480 | 0.95282 |
| 0.7 | 0.952452 | 0.95285 |
| 0.5 | 0.952368 | 0.95283 |

The cross-validated AUC **falls** as Part 9 comes in, and the public score rises. Part 9 has no published out-of-fold predictions, so the 0.95300 mix cannot be scored on the training rows.

**How we test it.** Our notebook has the same two kinds of stack: the 60-member stacker (our own members and the v16-v21 banks, the role of Part 9) and the wide stacker (those 60 plus 107 public members, the role of Part 12).
The wide stack already contains every member of the narrow one, so a mix can only help if the narrow stacker, with fewer columns and less regularisation, ranks some rows better.
Both stackers are held out on the same three outer splits (7, 8, 9), so their mix is scored on rows neither fitted. The cell prints the AUC for wide-stack weights 1, 0.9, 0.8, 0.65, 0.5 and 0 (the narrow stack alone).

**The test side.** The cell also prints how far apart the test files are (1 - rank correlation, in 1e-4): our wide and narrow stacks, Part 9, Part 12, and both 65/35 mixes.
goodpjw2008 measured their wide stack 95.0e-4 from Part 9 and called it "an unusually large distance between strong files"; the same number for our two stacks says whether our narrow stack can play Part 9's role at all.

**What is submitted.** `submission.csv` stays the wide stacker, unless the 65/35 mix of our two stacks beats it on all three splits (the weight is fixed at the public recipe's 0.65, not picked on these splits).
Two more files are written for the public board, where the recipe's evidence lives: `submission_mix_ours.csv` (65% our wide + 35% our narrow) and `submission_mix_part9.csv` (65% our wide + 35% goodpjw2008's Part 9, i.e. the recipe with our wide stack in Part 12's place).

**Credit.** goodpjw2008 for both stacks, the frozen files and the honest CV-vs-LB table; malyshevdanil for the 65/35 rank-logit recipe; areebbb and lucifer19 for the public-board probes around it. Please upvote them.
"""),
code("""# v25: the two-stack mix (malyshevdanil's 65/35 rank-logit recipe), scored on held-out rows of our own two stackers
import hashlib
def _lr(v): return logit(np.clip((rankdata(v) - 0.5) / len(v), 1e-7, 1 - 1e-7))
_sp = (7, 8, 9)
mix = {w: [roc_auc_score(y, w * _lr(wide_hold[s]) + (1 - w) * _lr(st_hold[s])) for s in _sp] for w in (1.0, 0.9, 0.8, 0.65, 0.5)}
mix[0.0] = [roc_auc_score(y, st_hold[s]) for s in _sp]
mt = pd.DataFrame(mix, index=[f'split {s}' for s in _sp]).T.sort_index(ascending=False); mt['mean'] = mt.mean(1)
mt['vs wide alone (x 1e-4)'] = (mt['mean'] - mt.loc[1.0, 'mean']) * 1e4
mt.index.name = 'weight of the wide stack'; display(mt.round(5))
print('rank correlation of the two held-out stacks per split:', [round(np.corrcoef(rankdata(wide_hold[s]), rankdata(st_hold[s]))[0, 1], 5) for s in _sp])
t_w, t_n = wstk.decision_function(ZWT), stk.decision_function(ZT)
T = {'our wide': t_w, 'our narrow (60)': t_n}
_h = {'part9_test.npy': 'e899205391b8755e82a2ae6b06fb1e026ef29132e4bcdbfad96535d7f5608abd', 'wide_lm_stack_test.npy': 'b37471e6c34a50715284e807d68e15e5f3ad033e72400904f27876b24041f8cc'}
for f_, lab in [('part9_test.npy', 'goodpjw2008 Part 9'), ('wide_lm_stack_test.npy', 'goodpjw2008 Part 12 wide')]:
    p_ = _find(f_, 'fraud-series-artifacts')
    if p_ is None: print(f_, 'not mounted, skipped'); continue
    hx = hashlib.sha256(open(p_, 'rb').read()).hexdigest()
    print(f_, 'SHA-256 matches the 0.95300 notebooks' if hx == _h[f_] else f'SHA-256 CHANGED: {hx}')
    T[lab] = np.load(p_).ravel(); assert len(T[lab]) == len(XT)
T['mix 0.65 our wide + 0.35 our narrow'] = 0.65 * _lr(t_w) + 0.35 * _lr(t_n)
if 'goodpjw2008 Part 9' in T:
    T['mix 0.65 our wide + 0.35 Part 9'] = 0.65 * _lr(t_w) + 0.35 * _lr(T['goodpjw2008 Part 9'])
if 'goodpjw2008 Part 9' in T and 'goodpjw2008 Part 12 wide' in T:
    T['mix 0.65 Part 12 + 0.35 Part 9 (LB 0.95300)'] = 0.65 * _lr(T['goodpjw2008 Part 12 wide']) + 0.35 * _lr(T['goodpjw2008 Part 9'])
TR = np.corrcoef([rankdata(v) for v in T.values()])
print('test files: distance 1 - rank correlation, x 1e-4'); display(pd.DataFrame((1 - TR) * 1e4, index=list(T), columns=list(T)).round(1))
fig, a1 = plt.subplots(figsize=(8, 4))
for j, s in enumerate(_sp): a1.plot(mt.index, mt[f'split {s}'], 'o-', alpha=0.5, label=f'split {s}')
a1.plot(mt.index, mt['mean'], 'k-', lw=2, label='mean'); a1.axvline(0.65, color='#c98b2b', ls='--', lw=1, label='public recipe 0.65')
a1.set_xlabel('weight of the wide stack (rest: the 60-member stack)'); a1.set_ylabel('held-out AUC'); a1.set_ylim(mt['mean'].max() - 0.0004, mt['mean'].max() + 0.0001)
a1.set_title('the two-stack mix on held-out rows'); a1.legend(fontsize=8); plt.tight_layout(); plt.show()
for n_, v_ in [('submission_mix_ours.csv', T['mix 0.65 our wide + 0.35 our narrow']), ('submission_mix_part9.csv', T.get('mix 0.65 our wide + 0.35 Part 9'))]:
    if v_ is not None: pd.DataFrame({'id': test['id'], TARGET: rankdata(v_) / len(v_)}).to_csv(OUT / n_, index=False); print('wrote', n_)
if np.mean(wide_runs) > np.mean(st_runs) and all(a > b for a, b in zip(mix[0.65], mix[1.0])):
    v_ = T['mix 0.65 our wide + 0.35 our narrow']
    pd.DataFrame({'id': test['id'], TARGET: rankdata(v_) / len(v_)}).to_csv(OUT / 'submission.csv', index=False)
    print('the 65/35 mix beats the wide stacker on all three splits: submission.csv is the mix')
else:
    print('the 65/35 mix does not beat the wide stacker on all three splits: submission.csv stays as written above')
"""),
md("""### What the board said about v25 (submitted 2026-11-09)

| file | what | public LB |
|---|---|---|
| `submission.csv` | our wide stacker (v23's ranking, unchanged) | 0.95260 (v23) |
| `submission_mix_part9.csv` | 0.65 our wide + 0.35 goodpjw2008 Part 9 | 0.95259 |
| `submission_mix_ours.csv` | 0.65 our wide + 0.35 our 60-member stack | 0.95244 |

Held-out AUC (splits 7 / 8 / 9): wide 0.95257 / 0.95259 / 0.95255, the 65/35 mix of our two stacks 0.95255 / 0.95259 / 0.95255.
Both say the same: mixing a narrow stack into our wide one does not help. Part 9 on top of our wide stack does nothing on the board (-0.00001), and our own narrow stack costs 0.00016.
So the 0.95300 is not "a wide stack plus a narrow stack" in general. On the test rows our wide stack is already 25e-4 from Part 9, while goodpjw2008's Part 12 is 83e-4 from both.
The next question is which half of the 0.95300 mix carries the gain. The section below tests it.

### v26: the 0.95330 file, and how large a leaderboard step is

**The question.** On 2026-11-09 the public best moved from 0.95300 to **0.95330**: [Moran Z's Grandmaster Ladder](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330) published a frozen file, `submission_095330.csv` ("Strike 9"),
and [areebbb's 0.95330 Ladder](https://www.kaggle.com/code/areebbb/s6e11-the-0-95330-ladder-what-actually-moved-lb) copies it with CV checks around it.
The file has no out-of-fold predictions. Moran Z's ladder lists ten public-board steps from the Part 12 backbone (0.95307) to 0.95330; four of them are "LB-gradient ascent" or "continue the high-slope axis",
i.e. steps chosen by their public score. From 0.95319 upward each step is +0.00001 to +0.00003. How large is that next to the noise of the public board, and how far is the file from ours?

**What the cell measures.**
1. **Distance on the test rows** (1 - rank correlation, in 1e-4) between our wide stack, goodpjw2008's Part 9 and Part 12, the 0.95300 mix, the 0.95330 file, and one new mix.
2. **The new mix**, `submission_p12_ours.csv`: 0.65 goodpjw2008 Part 12 + 0.35 **our** wide stack, i.e. the 0.95300 recipe with our wide stack in Part 9's place.
   v25 put our stack in Part 12's place and scored 0.95259. If this one scores near 0.95300, the gain of the recipe lives in Part 12's test predictions, and the 35% partner can be any good Part 9-like file.
3. **The noise of a public-board step.** The public board scores about 20% of the test rows. We draw 200 random subsets of that size from our held-out rows (split 7)
   and, for pairs of files at a known distance, measure the spread of their AUC *difference*. The same for 80%-sized subsets, the size of the private board.
   If the spread of a difference is larger than a ladder step, a step picked on the public board is mostly a draw of that noise, and it need not carry to the private board.

**Credit.** Moran Z for the 0.95330 file and the ladder; areebbb for the copy, the CV checks and the note that late steps are the size of the noise; goodpjw2008 for Part 9 and Part 12. Please upvote them.
"""),
code("""# v26: the 0.95330 file (Moran Z's Strike 9) on the test side, our wide stack in Part 9's place, and the noise of a public-board step
p_ = _find('submission_095330.csv', 'strike9')
if p_ is None: print('submission_095330.csv not mounted, skipped')
else:
    hx = hashlib.sha256(open(p_, 'rb').read()).hexdigest()
    print('submission_095330.csv', 'SHA-256 matches Moran Z' if hx == 'f4a7a4f8636c99fabd82945c7d81956c5559c609dcb55cde6d4e03f80b0ea105' else f'SHA-256 CHANGED: {hx}')
    s9_ = pd.read_csv(p_); assert (s9_['id'].to_numpy() == test['id'].to_numpy()).all()
    T['Moran Z Strike 9 (LB 0.95330)'] = s9_[TARGET].to_numpy()
if 'goodpjw2008 Part 12 wide' in T:
    T['mix 0.65 Part 12 + 0.35 our wide'] = 0.65 * _lr(T['goodpjw2008 Part 12 wide']) + 0.35 * _lr(t_w)
    v_ = T['mix 0.65 Part 12 + 0.35 our wide']
    pd.DataFrame({'id': test['id'], TARGET: rankdata(v_) / len(v_)}).to_csv(OUT / 'submission_p12_ours.csv', index=False); print('wrote submission_p12_ours.csv')
k26 = [k for k in ['our wide', 'goodpjw2008 Part 9', 'goodpjw2008 Part 12 wide', 'mix 0.65 Part 12 + 0.35 Part 9 (LB 0.95300)', 'mix 0.65 Part 12 + 0.35 our wide', 'Moran Z Strike 9 (LB 0.95330)'] if k in T]
D26 = pd.DataFrame((1 - np.corrcoef([rankdata(T[k]) for k in k26])) * 1e4, index=k26, columns=[str(i) for i in range(len(k26))])
print('test files: distance 1 - rank correlation, x 1e-4 (columns in row order)'); print(D26.round(1).to_string())
# the noise of a step: AUC differences of two held-out files on random public-sized (20%) and private-sized (80%) subsets of split 7's rows
yv, rng26 = y.to_numpy(), np.random.default_rng(26)
a_ = wide_hold[7]
pairs26 = {'wide vs 0.9 wide + 0.1 narrow': 0.9 * _lr(a_) + 0.1 * _lr(st_hold[7]), 'wide vs 0.65 wide + 0.35 narrow': 0.65 * _lr(a_) + 0.35 * _lr(st_hold[7]), 'wide vs narrow (60)': st_hold[7]}
rows26, draws26 = [], {}
for lab, b_ in pairs26.items():
    full = (roc_auc_score(yv, a_) - roc_auc_score(yv, b_)) * 1e4
    r_ = [lab, (1 - np.corrcoef(rankdata(a_), rankdata(b_))[0, 1]) * 1e4, full]
    for frac in (0.2, 0.8):
        n_ = int(frac * len(XT)); d_ = []
        for _ in range(200):
            i_ = rng26.choice(len(yv), n_, replace=False); d_.append((roc_auc_score(yv[i_], a_[i_]) - roc_auc_score(yv[i_], b_[i_])) * 1e4)
        draws26[(lab, frac)] = np.array(d_); r_ += [np.std(d_), np.mean(np.sign(d_) != np.sign(full))]
    rows26.append(r_)
N26 = pd.DataFrame(rows26, columns=['pair', 'distance x 1e-4', 'AUC difference, all rows x 1e-4', 'spread on 20% (public) x 1e-4', 'share of 20% draws with the sign flipped',
                                    'spread on 80% (private) x 1e-4', 'share of 80% draws with the sign flipped']).set_index('pair')
print(f'noise of an AUC difference, 200 random subsets of split 7 held-out rows ({int(0.2 * len(XT))} and {int(0.8 * len(XT))} rows)'); print(N26.round(3).to_string())
_ss = [np.std([roc_auc_score(yv[i_], a_[i_]) for i_ in [rng26.choice(len(yv), int(0.2 * len(XT)), replace=False) for _ in range(100)]]) * 1e4]
print(f'spread of one file\\'s AUC on public-sized subsets: {_ss[0]:.1f}e-4')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 1.2]})
im = a1.imshow(D26.values, cmap='Blues'); a1.set_xticks(range(len(k26))); a1.set_yticks(range(len(k26)))
_short = [k.replace('goodpjw2008 ', '').replace(' (LB 0.95300)', '\\n(0.95300)').replace(' (LB 0.95330)', '\\n(0.95330)').replace('mix 0.65 ', '').replace(' + 0.35 ', ' / ') for k in k26]
a1.set_xticklabels(_short, rotation=60, ha='right', fontsize=7); a1.set_yticklabels(_short, fontsize=7)
for i in range(len(k26)):
    for j in range(len(k26)): a1.text(j, i, f'{D26.values[i, j]:.0f}', ha='center', va='center', fontsize=7, color='k' if D26.values[i, j] < 60 else 'w')
a1.set_title('test files: 1 - rank correlation (x 1e-4)', fontsize=10)
_lab = 'wide vs 0.65 wide + 0.35 narrow'
for frac, col in [(0.2, '#c98b2b'), (0.8, '#4a6fa5')]: a2.hist(draws26[(_lab, frac)], bins=30, alpha=0.6, color=col, label=f'{int(frac * 100)}% of the test size')
a2.axvline(N26.loc[_lab, 'AUC difference, all rows x 1e-4'], color='k', lw=1.5, label='all held-out rows')
for s_ in (0.1, 0.3): a2.axvspan(-s_, s_, color='#999', alpha=0.15)
a2.set_xlabel('AUC(wide) - AUC(65/35 mix), x 1e-4  (grey: a ladder step of 0.00001 / 0.00003)'); a2.set_title(f'the same two files, scored on random subsets: {_lab}', fontsize=10); a2.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
md("""### What the board said about v26 (submitted 2026-11-10)

| file | what | public LB |
|---|---|---|
| `submission.csv` | our wide stacker (v23's ranking, unchanged) | 0.95260 (v23) |
| `submission_p12_ours.csv` | 0.65 goodpjw2008 Part 12 + 0.35 our wide stack | **0.95305** |
| (public) | 0.65 Part 12 + 0.35 Part 9, the 0.95300 recipe | 0.95300 |
| `submission_mix_part9.csv` (v25) | 0.65 our wide + 0.35 Part 9 | 0.95259 |

With our stack in Part 9's place the recipe scores 0.95305; with our stack in Part 12's place it scored 0.95259. So the recipe's gain comes with goodpjw2008's Part 12 test file, and the 35% partner can be Part 9 or our stack.
Our wide stack and Part 12 have the same held-out AUC (0.95257 both), so the gap is on the test side. HYPOTHESIS (unverified): Part 12's test columns were made differently (full-data refits against fold averages).
0.95305 against 0.95300 is +0.00005 between files 3.1e-4 apart; the noise table of v26 gives a spread of 0.29e-4 to 0.87e-4 for such pairs on the public rows. Treat the two as equal.

**The noise of a public-board difference, as v26 measured it** (200 random subsets of held-out rows; values x 1e-4):

| pair | distance | AUC difference, all rows | spread on public-sized subsets | spread on private-sized subsets |
|---|---|---|---|---|
| wide vs 0.9 wide + 0.1 narrow | 0.33 | -0.09 | 0.29 | 0.11 |
| wide vs 0.65 wide + 0.35 narrow | 4.0 | 0.01 | 0.87 | 0.39 |
| wide vs narrow (60 members) | 32.9 | 2.40 | 2.80 | 1.08 |

The noise of a *difference* grows with the distance between the two files. A ladder step of 0.00001 between files 0.3e-4 apart is a third of one spread.

### v27: the public ladder, replayed on held-out rows

**The question.** On 2026-11-10 and 11 the public best moved from 0.95330 to **0.95347** in six steps:
[amanatar's Podium Apex](https://www.kaggle.com/code/amanatar/s6e11-podium-apex-public-lb-0-95340) ("Strike 10", 0.95340, a frozen file),
[goodpjw2008's LM Ladder Part 1](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346) (0.95346) and
[areebbb's Three Ladders, Two Steps](https://www.kaggle.com/code/areebbb/s6e11-three-ladders-two-steps-lb-0-95347) (0.95347).
All three say plainly that each step was chosen by its public score and has no out-of-fold check. areebbb's final recipe, in logit ranks, with `F` = Strike 9 and `A` = Strike 10:

```
B     = A + 0.6 (A - F) + 0.12 (LM - F) + 0.06 (L7 - F)        LM: megayak's language model, L7: goodpjw2008's own seed-7 language model
score = B - 0.05 (MLP - F) - 0.025 (GB - F)                    MLP, GB: two members of goodpjw2008's OOF bank
```

A step is `file + w (member - base)`: move the ranking a little towards one member (or away from it, with a negative `w`), try it on the board, keep it if the score went up.
The files `F` and `A` have no out-of-fold predictions, so nobody can score these steps on labelled rows. But the *method* can be replayed where the labels are known.

**How the replay works.**
1. The base is our wide stacker's held-out score on all 700k training rows (split 7): each row is scored by a stacker that did not see it. This plays the role of `F`.
2. The step directions are the four members the public ladders use, as out-of-fold predictions: megayak's language model, masayakawamata's GPT-2, and goodpjw2008's MLP bag (`gbn_bag5_w512`) and GB bag (`gb_bag5`).
3. One draw: pick 299,984 rows at random (the size of the test set); the first 20% are the "public board", the other 80% the "private board".
4. Climb: try every direction at `w` = ±0.03, ±0.06, ±0.12 (24 candidate files), keep the one with the best *public* AUC if it beats the current file, and repeat up to six times, as a ladder does.
5. Score the final file on the private rows. 100 draws.

If the steps find something real that the stacker missed, the private rows gain too. If they fit the public rows' noise, the public gain is there by selection and the private rows gain nothing or lose.

**Measured before this version ran** (same code on the published predictions, `scratch/banks/ladder_1111/sim.py` in github.com/alexchilton/kaggley; the cell below prints its own numbers):

| | AUC change x 1e-4 |
|---|---|
| public rows, mean over 100 draws (the steps were chosen here) | **+2.2** |
| private rows, mean over 100 draws | **-1.6** |
| share of draws where the private rows gained | 8% |
| correlation of public gain and private gain over the draws | -0.66 |

The climb gains +0.00022 on its public rows and loses 0.00016 on the private rows; the larger the public gain of a draw, the larger its private loss.
For scale: the public ladder climbed +0.00017 (0.95330 to 0.95347). A gain of that size appears from selection alone.

**One step without any selection** (all 700k held-out rows; AUC change x 1e-4):

| step | megayak LM | masayakawamata GPT-2 | MLP bag | GB bag |
|---|---|---|---|---|
| w = -0.12 | -0.03 | +0.16 | -0.63 | -0.55 |
| w = -0.06 | +0.05 | +0.15 | -0.26 | -0.23 |
| w = +0.06 | -0.19 | -0.30 | +0.14 | +0.13 |
| w = +0.12 | -0.51 | -0.75 | +0.15 | +0.17 |

On held-out rows, a step **towards** megayak's LM loses, and a step **away** from the MLP and GB bags loses. The public ladders took exactly these steps (+0.12 LM, -0.05 MLP, -0.025 GB) and gained on the board.
areebbb measured the same disagreement on goodpjw2008's nested OOF (full tree move -7.4e-5, +5% LM -1.5e-5).

**What this does and does not show.** It shows that the ladder *method*, started from a fitted stack, produces public gains of the size seen on the board with a private loss behind them.
It does not prove the real files will fall: the replay's base is a stacker that is already at its held-out optimum, so every direction is downhill, and the test-side predictions of a member are not the same thing as its out-of-fold predictions
(megayak's test file averages five fold models; an out-of-fold row is scored by one). HYPOTHESIS (unverified): part of the board's taste for the LM is that its test file is stronger than its OOF column says.
The private board will say. For your two final picks, the advice in areebbb's and goodpjw2008's notebooks holds: one best-public file and one file with an out-of-fold score.

**The test side, and two files for the board.** The cell rebuilds the 0.95347 file from the six public inputs (each checked by SHA-256) and prints the distance table with Strike 10, the 0.95347 file and our files.
It also takes the ladder's most-used step from our own files and writes them, because for these we have the held-out number above to put next to the board's:
- `submission_wide_lm12.csv`: our wide stack + 0.12 (megayak LM - our wide stack). Held-out: **-0.51e-4**. If the board gains, the board and the held-out rows disagree about the same step on the same stack.
- `submission_p12_ours_lm12.csv`: the same step from `submission_p12_ours.csv` (0.95305).
It also splits each ladder file's move from Strike 9 into its member directions by least squares, as areebbb did for Strike 10.

**Credit.** amanatar (Strike 10), goodpjw2008 (the LM ladder, the seed-7 language model, the OOF bank), areebbb (the 0.95347 recipe, every probe published including the losing ones, and the CV checks), Moran Z (Strike 9), megayak (the language model), samanyu1808 (the LM-step probes). Please upvote them.
"""),
code("""# v27: the public ladder replayed on held-out rows (public-sized subset picks the steps, private-sized subset scores them), and the ladder files on the test side
def _fauc(yy, s):                                         # AUC from ranks: the same number as roc_auc_score, faster
    r = rankdata(s); n1 = yy.sum(); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))
yv27 = y.to_numpy().astype(int); F27 = _lr(wide_hold[7])
_d27 = {'megayak LM': 'megayak target-first LM', 'masayakawamata GPT-2': 'masayakawamata GPT-2 target-first', 'goodpjw2008 MLP bag': 'goodpjw2008 gbn_bag5_w512', 'goodpjw2008 GB bag': 'goodpjw2008 gb_bag5'}
DIR27 = {k: _lr(pool[v][0]) for k, v in _d27.items() if v in pool}
print('step directions found in the pool:', list(DIR27))
a27 = _fauc(yv27, F27); WS27 = (-0.12, -0.06, -0.03, 0.03, 0.06, 0.12)
one27 = pd.DataFrame({k: [(_fauc(yv27, F27 + w * (d_ - F27)) - a27) * 1e4 for w in WS27] for k, d_ in DIR27.items()}, index=WS27); one27.index.name = 'w'
print(f'held-out AUC of the base (wide stacker, split 7): {a27:.5f}'); print('one step on all held-out rows, AUC change x 1e-4 (no selection):'); print(one27.round(3).to_string())
rng27, n_t, n_pub, rows27 = np.random.default_rng(27), len(XT), int(0.2 * len(XT)), []
for it in range(100):
    i_ = rng27.choice(len(yv27), n_t, replace=False); pu, pr = i_[:n_pub], i_[n_pub:]
    cur = np.zeros(len(yv27)); p0 = _fauc(yv27[pu], F27[pu]); best = p0; path = []
    for rnd in range(6):
        a_, k_, w_ = max((_fauc(yv27[pu], (F27 + cur + w * (d_ - F27))[pu]), k, w) for k, d_ in DIR27.items() for w in WS27)
        if a_ <= best: break
        best = a_; cur = cur + w_ * (DIR27[k_] - F27); path.append((k_, w_))
    rows27.append([len(path), (best - p0) * 1e4, (_fauc(yv27[pr], (F27 + cur)[pr]) - _fauc(yv27[pr], F27[pr])) * 1e4, path[0][0] if path else '-'])
S27 = pd.DataFrame(rows27, columns=['steps', 'public gain x 1e-4', 'private gain x 1e-4', 'first step'])
print(f'ladder replay, 100 draws of {n_t} held-out rows ({n_pub} public, {n_t - n_pub} private):'); print(S27.describe().round(3).to_string())
print(f"mean public gain {S27['public gain x 1e-4'].mean():+.2f}e-4 | mean private gain {S27['private gain x 1e-4'].mean():+.2f}e-4 | share of draws with a private gain {(S27['private gain x 1e-4'] > 0).mean():.2f} | "
      f"correlation {np.corrcoef(S27['public gain x 1e-4'], S27['private gain x 1e-4'])[0, 1]:.3f} | first step: {S27['first step'].value_counts().to_dict()}")
# the test side: Strike 10, the 0.95347 recipe rebuilt from its public inputs, and the LM step on our own files
_h27 = {'submission_095340.csv': ('strike10', '12b525b84f29e32d6511e9d6105f6dc5eac79ae41b10140eed3353ea57afc159'), 'lm7_test_f0.npy': ('compact-lm', '78af7498ee398b300556a8afc9d661fdfd2b42d8d632e6af9cd412aa3430a369'),
        'lm_test.npy': ('target-first-lm', '3168779be644a530c6bdc4c141cae168bfc2feb01ac21090c917cc978a7687d3')}
P27 = {}
for f_, (own_, h_) in _h27.items():
    p_ = _find(f_, own_)
    if p_ is None: print(f_, 'not mounted, skipped'); continue
    hx = hashlib.sha256(open(p_, 'rb').read()).hexdigest(); print(f_, 'SHA-256 matches the 0.95347 notebook' if hx == h_ else f'SHA-256 CHANGED: {hx}'); P27[f_] = p_
LMt = _lr(pool['megayak target-first LM'][1]) if 'megayak target-first LM' in pool else None
if LMt is not None:
    T['our wide + 0.12 (LM - our wide)'] = _lr(t_w) + 0.12 * (LMt - _lr(t_w))
    if 'mix 0.65 Part 12 + 0.35 our wide' in T:
        m_ = _lr(T['mix 0.65 Part 12 + 0.35 our wide']); T['P12/our wide mix + 0.12 (LM - mix)'] = m_ + 0.12 * (LMt - m_)
    for n_, k_ in [('submission_wide_lm12.csv', 'our wide + 0.12 (LM - our wide)'), ('submission_p12_ours_lm12.csv', 'P12/our wide mix + 0.12 (LM - mix)')]:
        if k_ in T: pd.DataFrame({'id': test['id'], TARGET: rankdata(T[k_]) / len(T[k_])}).to_csv(OUT / n_, index=False); print('wrote', n_)
S9_ = 'Moran Z Strike 9 (LB 0.95330)'
if 'submission_095340.csv' in P27 and S9_ in T:
    s10_ = pd.read_csv(P27['submission_095340.csv']); assert (s10_['id'].to_numpy() == test['id'].to_numpy()).all()
    A27, F9 = _lr(s10_[TARGET].to_numpy()), _lr(T[S9_]); T['amanatar Strike 10 (LB 0.95340)'] = A27
    _need = ['goodpjw2008 gbn_bag5_w512', 'goodpjw2008 gb_bag5']
    if 'lm7_test_f0.npy' in P27 and LMt is not None and all(k in pool for k in _need):
        a7 = np.load(P27['lm7_test_f0.npy']); L7 = _lr((a7[..., 1] - a7[..., 0]).mean(1)); MLPt, GBt = _lr(pool[_need[0]][1]), _lr(pool[_need[1]][1])
        G27 = A27 + 0.6 * (A27 - F9) + 0.12 * (LMt - F9) + 0.06 * (L7 - F9) - 0.05 * (MLPt - F9) - 0.025 * (GBt - F9); T['areebbb ladder, rebuilt (LB 0.95347)'] = G27
        Xd = np.column_stack([LMt - F9, L7 - F9, MLPt - F9, GBt - F9, _lr(t_w) - F9])
        for n_, v_ in [('Strike 10 - Strike 9', A27 - F9), ('0.95347 file - Strike 9', G27 - F9)]:
            c_ = np.linalg.lstsq(Xd, v_, rcond=None)[0]
            print(f'{n_}, least squares on member directions:', dict(zip(['megayak LM', 'LM seed 7', 'MLP bag', 'GB bag', 'our wide'], np.round(c_, 3))), '| share of the move explained', round(1 - ((v_ - Xd @ c_) ** 2).sum() / (v_ ** 2).sum(), 3))
k27 = [k for k in ['our wide', 'our wide + 0.12 (LM - our wide)', 'mix 0.65 Part 12 + 0.35 our wide', 'P12/our wide mix + 0.12 (LM - mix)', S9_, 'amanatar Strike 10 (LB 0.95340)', 'areebbb ladder, rebuilt (LB 0.95347)'] if k in T]
D27 = pd.DataFrame((1 - np.corrcoef([rankdata(T[k]) for k in k27])) * 1e4, index=k27, columns=[str(i) for i in range(len(k27))])
print('test files: distance 1 - rank correlation, x 1e-4 (columns in row order)'); print(D27.round(1).to_string())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
a1.scatter(S27['public gain x 1e-4'], S27['private gain x 1e-4'], s=18, color='#4a6fa5', alpha=0.7); a1.axhline(0, color='k', lw=0.8); a1.axvline(0, color='k', lw=0.8)
a1.axvline(1.7, color='#c98b2b', ls='--', lw=1, label='the public ladder: 0.95330 -> 0.95347')
a1.set_xlabel('gain on the public-sized rows that picked the steps (x 1e-4)'); a1.set_ylabel('gain on the private-sized rows (x 1e-4)')
a1.set_title('a ladder climbed on held-out rows: 100 draws', fontsize=10); a1.legend(fontsize=8)
for k in one27.columns: a2.plot(one27.index, one27[k], 'o-', label=k)
a2.axhline(0, color='k', lw=0.8); a2.set_xlabel('step w:  file + w (member - file)'); a2.set_ylabel('held-out AUC change, all rows (x 1e-4)')
a2.set_title('one step, no selection: which way is uphill on held-out rows', fontsize=10); a2.legend(fontsize=8); plt.tight_layout(); plt.show()
"""),
md("""### What the board said about v27 (submitted 2026-11-12)

| file | what | public LB | the same file without the step |
|---|---|---|---|
| `submission_p12_ours_lm12.csv` | (0.65 Part 12 + 0.35 our wide stack) + 0.12 (megayak LM - file) | **0.95305** | 0.95305 |
| `submission_wide_lm12.csv` | our wide stack + 0.12 (megayak LM - stack) | 0.95260 | 0.95260 |

The ladder's most-used step changes neither of our files at five decimals. On held-out rows the same step measured -0.51e-4.
The public ladders report +0.00001 to +0.00003 for it on their files. Our files already hold megayak's LM as a stacked member (since v23), at the weight the labelled rows chose.
One board step of 0.00001 is below the noise v26 measured, so "no change" and "+0.00001" cannot be told apart here. What the board does not show is a gain.

### v28: the 0.95370 file, taken apart and replayed on labelled rows

**The question.** On 2026-11-13 the public best is **0.95370**: a file by [Aman Atar](https://www.kaggle.com/datasets/amanatar/s6e11-rank1-q17-meta-bundle), which
[goodpjw2008's LM Ladder | Public vs CV Check](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-public-vs-cv-check-lb-0-95370) starts from.
Our best is 0.95305, and our own stack 0.95260. The file has no out-of-fold predictions. What is in it, and what would its mix score on rows with labels?

**The method (goodpjw2008's, run here against a second referee).** Aman Atar's bundle ships the file together with the test *and* out-of-fold predictions of 16 public models
(megayak's and masayakawamata's language models, goodpjw2008's two new seeds, two TabPFN runs, yekenot's RealMLP, sadamtorres' GBDTs, goodpjw2008's MLP and GB bags, sameershaikh27's ten-fold GLM boost, daoviet's GP + DVAE, megayak's shallow trees, and `alex_glm_gen`, a GLM member of this notebook).
1. Turn everything into logit ranks. Fit the file by least squares on the 16 models' **test** predictions. The weights say what the file is made of.
2. Apply the same weights to the 16 models' **out-of-fold** predictions. This is the file's mix on rows with labels: the replay.
3. Compare the replay with a stack that was fitted on labels. goodpjw2008 used their wide LM stack (nested CV 0.95257). We use ours: the wide stacker's held-out score on split 7, where every row is scored by a stacker that did not see it.
   The two stacks were built by different people from overlapping public members, so this is a second measurement of the same gap, not a copy of the first.

**What we measured** (same code on the published files before this version ran, `scratch/banks/v28_1113/sim28.py` in github.com/alexchilton/kaggley; the cell below prints its own numbers; 1 point = 0.00001 AUC):

| | AUC on labelled rows | public LB |
|---|---|---|
| the 0.95370 file (its mix replayed on out-of-fold rows) | 0.95186 | **0.95370** |
| 90% file + 10% our wide stack | 0.95200 (+13.4 points, 5 of 5 folds) | this version writes it: `submission_a370_wide10.csv` |
| 75% file + 25% our wide stack | 0.95217 (+30.9, 5 of 5) | `submission_a370_wide25.csv` |
| 50% / 50% | 0.95239 (+52.8, 5 of 5) | |
| our wide stack alone (held-out) | **0.95257** (+70.1, 5 of 5) | 0.95260 |

- The 16 models explain **98.3%** of the file on the test rows. The replay scores **0.95186**, the number goodpjw2008's notebook prints for the same step.
- Our held-out stack is **70 points above the replay**. On random subsets of the labelled rows the gap is -72 ± 37 points at the public board's size (59,996 rows; the file is ahead in 4 of 100 draws) and -70 ± 15 at the private board's size (239,987 rows; 0 of 100).
  On the public board the file is 110 points *above* our stack (0.95370 against 0.95260). goodpjw2008 measured -67 ± 34 against their stack. Two referees, one answer: the order on labelled rows is the reverse of the order on the public board, and sampling noise of the public rows does not cover the difference.
- **What the file is made of, and what the labels would choose for the same 16 models.** The cell fits a second set of weights on the same 16 out-of-fold columns, this time on the labels (logistic stacker, C 0.01, held out on the members' five folds).
  That stack scores **0.95211**: +25 points over the file's weights with the same members. The rest of the gap to our wide stack (+45) comes with the wider pool (167 members).
  Where the two sets of weights differ (file / labels, both scaled to the same sum): sadamtorres' XGBoost 0.67 / 0.16 and goodpjw2008's GB bag 0.61 / 0.18 (the file holds three to four times more);
  masayakawamata's GPT-2 0.06 / 0.18, its second seed -0.02 / 0.15, sameershaikh27's ten-fold GLM boost -0.04 / 0.21, megayak's shallow trees -0.19 / 0.14, sadamtorres' LightGBM -0.30 / 0.05 (the file holds much less, or subtracts).
  Where they agree: megayak's LM is large in both (0.48 / 0.18), and our GLM member is **subtracted in both** (-0.31 / -0.24). A negative weight is not by itself a mark of board fitting: the labels subtract that member too.
  The ten-fold GLM boost is the member v18 to v21 found raising our held-out AUC while lowering our public score (section 4). The file, tuned on the public board, drops it; the labels keep it. Same disagreement, seen from the other side.
- **More of a CV-backed stack helps at every share** on labelled rows. goodpjw2008 scored two such mixes on the board with their stack: 10% cost 2 public points (0.95368), 25% cost 8 (0.95362).

**What this does and does not show.** The replay is a stand-in: 1.7% of the file is not explained by the 16 models and is not replayed. The bundle also ships an `oof_anchor_095370` column; we do not use it, for goodpjw2008's reason:
it scores 0.95254 and has rank correlation 0.9987 with our held-out stack, while the replay of the file's own weights has 0.9943. It is the out-of-fold column of a wide stack, not of this file's weights.
This is not a verdict on the file or its author; every ladder on this board was built with public probes, and the authors publish them.
The practical reading is the one goodpjw2008 and areebbb give: for the two final picks, one best-public file and one file with a score on labelled rows. The two mix files below sit between them.

### v28: two more language-model seeds in the wide pool

**What it is.** goodpjw2008 trained a second seed of each public target-first language model and published the predictions on our folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, CC0):
[seed 7 of megayak's compact LM](https://www.kaggle.com/datasets/goodpjw2008/s6e11-compact-lm-seed7) (five folds, OOF 0.94945) and
[seed 2026 of masayakawamata's GPT-2](https://www.kaggle.com/datasets/goodpjw2008/s6e11-target-first-gpt2-seed-2026) (OOF on five folds 0.95014; the test predictions come from the fold-0 model only, the other four are still scoring).
v28 adds both to the wide pool as members 168 and 169 and refits the same logistic stacker (C 0.01).

**Why it should help.** The two first seeds hold the largest weights in our wide stack. A language model trained with another seed makes the same kind of prediction with different noise, so the pair is a less noisy version of one member.
The public ladders use them as a *rung*: `+ 0.06 (second seed - first seed)`, which moves 0.06 of the weight from one seed to the other. goodpjw2008 and areebbb call the GPT-2 rung "the first rung that gains on both boards" (+2 points public, +1.9e-5 on nested CV, 5 of 5 folds).
A stacker can do more than a rung: it picks each seed's weight on all 700k labelled rows.

**What we measured** (offline, same script):

| | split 7 | split 8 | split 9 |
|---|---|---|---|
| wide stacker, 167 members (held-out AUC) | 0.952566 | 0.952593 | 0.952552 |
| + both seeds, 169 members | 0.952587 | 0.952605 | 0.952572 |
| change, points | **+2.1** | **+1.2** | **+1.9** |
| + seed 7 of the compact LM only | +0.6 | | |
| + seed 2026 of the GPT-2 only | +1.5 | | |

| the public rung, no selection, all labelled rows | on our held-out wide stack | on the replayed 0.95370 file |
|---|---|---|
| + 0.06 (GPT-2 seed 2026 - masayakawamata's GPT-2) | +2.0 points, 5 of 5 folds | +0.2, 3 of 5 |
| + 0.06 (LM seed 7 - megayak's LM) | +1.3, 4 of 5 | +3.5, 5 of 5 |

- **A second seed is real information, and it is small.** Both seeds together add +1.2 to +2.1 points on three splits; every split is up. goodpjw2008's numbers for the rungs (+1.9 and +3.5) reproduce on our rows to the decimal for the file they were made on.
- **The stacker splits the weight by itself.** Before: masayakawamata's GPT-2 0.338, megayak's LM 0.299 (v27's log). With the seeds in: GPT-2 0.230 and its second seed 0.212, megayak's LM 0.232. The pair of GPT-2 seeds holds 0.44 where one seed held 0.34: two noisy copies are worth more weight than one.
- **The public board cannot see a gain of this size.** On random held-out subsets of the public board's size the change is +1.3 ± 7.6 points and positive in 60% of 200 draws; at the private board's size +2.6 ± 3.2, positive in 84%.
  A rung that "gained 2 points on the board" and a member that adds 2 points on 700k labelled rows are different kinds of evidence, even when the number is the same.
- **`submission.csv` does not change.** Our bar for a member to enter the submitted stack is +5 points (0.00005) on the mean of the three splits with every split up. +1.7 is below it.
  The 169-member stack is written as `submission_wide_seeds.csv` for the board; on the test rows it is 2.2e-4 from the wide stack (1 - rank correlation).
- A caveat on seed 2026: its test predictions come from one fold model. Its out-of-fold column is as good as the first seed's (0.95014 against 0.95003), but its test column is the noisier of the two until the other four fold models are published.

### v28: two checks on every member of the pool

A wide pool takes other people's files on trust. [dariushafshar's 145 OOF Pairs, One Stack](https://www.kaggle.com/code/dariushafshar/s6e11-145-oof-pairs-one-stack-lb-0-95246) runs checks on every pair before stacking it. Two of them, on our 169 columns:

**1. The fold signature.** Out-of-fold predictions come from five models, one per fold, and each model has its own offset. So the mean prediction differs between the five folds by more than chance allows, *if* the column was built on these folds.
`F` = the variance of the five fold means, divided by what random groups of the same size would give. About 1 means no signature. The control: the same statistic on a split no member was built on (seed 7).

- On our folds: median F **74**; on the control split: median 0.52, largest 2.3. The statistic separates the two cases cleanly.
- Six members have F below 5 on our folds: queyrusi's digit CNN bag (0.0), three of sadamtorres' v6 pairs (1.1 to 1.2), ektarr's RealMLP stack (1.2) and sadamtorres' raw LightGBM (3.6).
  sameershaikh27's ten-fold GLM boost has 5.0, as it should: ten folds, not our five. goodpjw2008's seed-7 LM has 6.0 and is on our folds by its README.
- **A low F is a question, not a verdict.** We looked at the lowest one. queyrusi's bank ships its fold ids, and they are our partition; its out-of-fold column has mean 0.5 and the same 10% and 90% quantiles in every fold: it was turned into ranks *inside each fold*, which removes the signature by construction.
  So F says "this column does not look like five raw fold models on these folds": other folds, a bag over several splits, or a per-fold rank transform. For the five others we have no fold ids to check, and their out-of-fold claim rests on the author's description.
  The stacker that weights them is itself held out, so a leak would show as a held-out AUC that the board does not confirm. v18 to v21 saw that pattern after the ten-fold member came in (section 4). HYPOTHESIS (unverified) that the two are linked.

**2. Is a member closer to the stack on the test rows than out of fold?** For each member: its rank correlation with the wide stack on out-of-fold rows, and the same on the test rows.
An out-of-fold row is scored by **one** fold model. A test row is usually scored by the **average of five** (or by a refit on all rows). An average is less noisy, so the test column sits closer to the stack than the OOF column.

- Nearly every member is closer to the stack on test. Medians by source: members trained here +0.0058, public banks +0.0035, the public pool +0.0038 (v28's text said +0.006, +0.005 and +0.004, written from a smoke run; corrected in v29 to the values the cell prints). The neural members move most: RealMLP on digits +0.038, RealMLP +0.032, TabM +0.021.
- The language models give a clean natural experiment. megayak's LM and seed 7 average five fold models on test: **+0.0024** and **+0.0035**. Seed 2026 scores the test rows with the fold-0 model only: **0.0000**. masayakawamata's GPT-2: +0.0005.
- Why it matters: the stacker learns each member's weight from its out-of-fold column, the noisy version. A member whose test column is five times averaged is better on test than the stacker was told, so its weight is too low for the test rows.
  This is a measured piece of v27's hypothesis for why the public board rewards "a step towards the LM" that held-out rows reject. It is not proof: the board gains of the ladders are also the size of selection noise (v27).

**Credit.** Aman Atar (the 0.95370 file and the bundle of 16 models), goodpjw2008 (the replay method, the board scores of the mixes, both new seeds), megayak, masayakawamata and Matapanino (the language-model code and first seeds), Chris Deotte (the target-first recipe),
dariushafshar (the pool checks), asteriospanoras ([How Much of the Public Top Is Real?](https://www.kaggle.com/code/asteriospanoras/s6e11-how-much-of-the-public-top-is-real): the noise of a public gap on out-of-fold rows), areebbb (the two-seed rung rebuilt and probed), and the authors of the 16 models in the bundle. Please upvote them.
"""),
code("""# v28: the public 0.95370 file taken apart and replayed on labelled rows, two more language-model seeds in the wide pool, and two checks on every pool member
from sklearn.linear_model import LinearRegression
_h28 = {'rank1_q17_streams.npz': ('rank1-q17', '0fe4ab1263a673c8065d7145590b92da440263161bf860ce45f2976b5c4fa56f'),
        'lm7_oof_5fold.npy': ('compact-lm', 'c56d6b7282e7ea8c4c5927ac27e8b286d539ffbd66d5795564323c8b92b65242'), 'lm7_test_5fold.npy': ('compact-lm', '68639fec79f081e97d40185443fa3903c9db3e62698abb01dfe31dcee217462a'),
        'gpt2_s2026_oof_5fold.npy': ('seed-2026', '4a902ecd3d725d21123f5b6239ac2190a9fd94f3de37e3157be391fcf3fe996a'), 'gpt2_s2026_test_f0.npy': ('seed-2026', '40e08058ed7632ee328c2fe7d97040b856b36d2636f32cc6d2781ec9db080695')}
P28 = {}
for f_, (own_, h_) in _h28.items():
    p_ = _find(f_, own_)
    if p_ is None: print(f_, 'not mounted, skipped'); continue
    hx = hashlib.sha256(open(p_, 'rb').read()).hexdigest(); print(f_, 'SHA-256 matches the 2026-11-13 download' if hx == h_ else f'SHA-256 CHANGED: {hx}'); P28[f_] = p_
yv28 = y.to_numpy().astype(int); F28 = _lr(wide_hold[7]); a28 = _fauc(yv28, F28); n_pub28, n_prv28 = int(0.2 * len(XT)), int(0.8 * len(XT))
def _ups28(v, b): return sum(_fauc(yv28[fold42 == k], v[fold42 == k]) > _fauc(yv28[fold42 == k], b[fold42 == k]) for k in range(5))
def _dist28(a, b): return (1 - np.corrcoef(rankdata(a), rankdata(b))[0, 1]) * 1e4

# 1. the 0.95370 file: least squares on the bundle's 16 public models (test rows), the same weights replayed on their out-of-fold predictions
W28 = M28 = AO28 = None
if 'rank1_q17_streams.npz' in P28:
    Zb = np.load(P28['rank1_q17_streams.npz'], allow_pickle=False); assert (Zb['y_train'] == yv28).all() and _same_folds(Zb['folds'])
    N28 = [str(s) for s in Zb['stream_names']]
    XO28 = np.column_stack([_lr(Zb['oof_' + n]) for n in N28]); XT28 = np.column_stack([_lr(Zb['test_' + n]) for n in N28]); AT28 = _lr(Zb['test_anchor_095370'])
    fit28 = LinearRegression().fit(XT28, AT28); AO28 = _lr(XO28 @ fit28.coef_)
    sl_ = np.zeros(len(yv28))                              # the same 16 models, weighted by the labels instead: our stacker's settings, held out on the members' five folds
    for k in range(5): sl_[fold42 == k] = LogisticRegression(C=0.01, max_iter=3000, tol=1e-6).fit(XO28[fold42 != k], yv28[fold42 != k]).decision_function(XO28[fold42 == k])
    cl_ = LogisticRegression(C=0.01, max_iter=3000, tol=1e-6).fit(XO28, yv28).coef_[0]; LAB28 = _fauc(yv28, sl_)
    W28 = pd.DataFrame({'weight in the file': fit28.coef_, 'weight the labels choose (same sum)': cl_ / cl_.sum() * fit28.coef_.sum(), 'OOF AUC alone': [_fauc(yv28, XO28[:, j]) for j in range(len(N28))]}, index=N28).sort_values('weight in the file', ascending=False)
    print(f'the 16 public models explain R2 = {fit28.score(XT28, AT28):.4f} of the 0.95370 file on the test rows'); print(W28.round({'weight in the file': 3, 'weight the labels choose (same sum)': 3, 'OOF AUC alone': 5}).to_string())
    print(f'the same 16 models with weights fitted on the labels (held out): AUC {LAB28:.5f}, {(LAB28 - _fauc(yv28, AO28)) * 1e5:+.1f} points against the weights of the file; correlation of the two weight vectors {np.corrcoef(fit28.coef_, cl_)[0, 1]:.2f}')
    print(f'the mix replayed on out-of-fold rows: AUC {_fauc(yv28, AO28):.5f} | our wide stacker, held-out (split 7): {a28:.5f} | gap {(_fauc(yv28, AO28) - a28) * 1e5:+.1f} points (1 point = 0.00001)')
    rng28 = np.random.default_rng(28); G28 = {}
    for n_ in (n_pub28, n_prv28):
        d_ = []
        for _ in range(100):
            i_ = rng28.choice(len(yv28), n_, replace=False); d_.append((_fauc(yv28[i_], AO28[i_]) - _fauc(yv28[i_], F28[i_])) * 1e5)
        G28[n_] = np.array(d_); print(f'gap (replayed file - our wide) on {n_} random rows: {np.mean(d_):+.1f} +- {np.std(d_):.1f} points; the file ahead in {np.mean(np.array(d_) > 0):.0%} of 100 draws')
    M28 = pd.DataFrame([[w, _fauc(yv28, (1 - w) * AO28 + w * F28), (_fauc(yv28, (1 - w) * AO28 + w * F28) - _fauc(yv28, AO28)) * 1e5, _ups28((1 - w) * AO28 + w * F28, AO28)] for w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)],
                       columns=['share of our wide stack', 'AUC on labelled rows', 'points vs the file', 'folds up of 5']).set_index('share of our wide stack')
    print('the replayed file mixed with our held-out wide stack:'); print(M28.round({'AUC on labelled rows': 5, 'points vs the file': 1}).to_string())
    T['amanatar 0.95370 file'] = AT28
    for w, n_ in [(0.10, 'submission_a370_wide10.csv'), (0.25, 'submission_a370_wide25.csv')]:
        T[f'{1 - w:.2f} file + {w:.2f} our wide'] = (1 - w) * AT28 + w * _lr(t_w); v_ = T[f'{1 - w:.2f} file + {w:.2f} our wide']
        pd.DataFrame({'id': test['id'], TARGET: rankdata(v_) / len(v_)}).to_csv(OUT / n_, index=False); print('wrote', n_)
    k28 = [k for k in ['our wide', 'mix 0.65 Part 12 + 0.35 our wide', 'areebbb ladder, rebuilt (LB 0.95347)', 'amanatar 0.95370 file', '0.90 file + 0.10 our wide', '0.75 file + 0.25 our wide'] if k in T]
    D28 = pd.DataFrame((1 - np.corrcoef([rankdata(T[k]) for k in k28])) * 1e4, index=k28, columns=[str(i) for i in range(len(k28))])
    print('test files: distance 1 - rank correlation, x 1e-4 (columns in row order)'); print(D28.round(1).to_string())
    print(f'labelled rows: distance of the replayed file from our held-out wide stack {_dist28(AO28, F28):.1f}e-4')

# 2. two more language-model seeds as members of the wide pool (goodpjw2008's seed-7 compact LM and seed-2026 GPT-2)
new28 = {}
if 'lm7_oof_5fold.npy' in P28 and 'lm7_test_5fold.npy' in P28: new28['goodpjw2008 LM seed 7'] = (np.load(P28['lm7_oof_5fold.npy']).astype(float).ravel(), np.load(P28['lm7_test_5fold.npy']).astype(float).ravel())
if 'gpt2_s2026_oof_5fold.npy' in P28 and 'gpt2_s2026_test_f0.npy' in P28: new28['goodpjw2008 GPT-2 seed 2026'] = (np.load(P28['gpt2_s2026_oof_5fold.npy']).astype(float).ravel(), np.load(P28['gpt2_s2026_test_f0.npy']).astype(float).ravel())
new28 = {k: v for k, v in new28.items() if len(v[0]) == len(X) and len(v[1]) == len(XT) and np.isfinite(v[0]).all() and np.isfinite(v[1]).all()}
_tw28 = {'goodpjw2008 LM seed 7': 'megayak target-first LM', 'goodpjw2008 GPT-2 seed 2026': 'masayakawamata GPT-2 target-first'}       # each new member and its first seed, already in the pool
for k, (o_, t_) in new28.items():
    print(f'{k}: OOF AUC {_fauc(yv28, o_):.5f}' + (f'; rank correlation with {_tw28[k]} {np.corrcoef(rankdata(o_), rankdata(pool[_tw28[k]][0]))[0, 1]:.4f} (OOF)' if _tw28[k] in pool else ''))
R28 = []
for k in new28:
    if _tw28[k] not in pool: continue
    d_ = 0.06 * (_lr(new28[k][0]) - _lr(pool[_tw28[k]][0]))
    for lab, b_ in [('our held-out wide stack', F28)] + ([('the replayed 0.95370 file', AO28)] if AO28 is not None else []):
        R28.append([f'+ 0.06 ({k} - {_tw28[k]})', lab, (_fauc(yv28, b_ + d_) - _fauc(yv28, b_)) * 1e5, _ups28(b_ + d_, b_)])
if R28: print('a rung of 0.06 from the first seed to the second, no selection, all labelled rows:'); print(pd.DataFrame(R28, columns=['rung', 'on', 'points', 'folds up of 5']).round(2).to_string(index=False))
w28_runs, w28_hold, wn28 = [], {}, wnames + list(new28)
if new28:
    ZW28 = np.column_stack([ZW] + [lr_(v[0]) for v in new28.values()]); ZWT28 = np.column_stack([ZWT] + [lr_(v[1]) for v in new28.values()])
    for seed in (7, 8, 9):
        s = np.zeros(len(ZW28))
        for va_i, dv in Parallel(n_jobs=-1)(delayed(_fold)(ZW28, y.values, tr_i, va_i) for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(ZW28, y)): s[va_i] = dv
        w28_runs.append(roc_auc_score(y, s)); w28_hold[seed] = s
    print(f'held-out AUC over splits 7, 8, 9: wide stacker ({len(wnames)} members) {np.mean(wide_runs):.5f} {np.round(wide_runs, 5)} | with the two seeds ({len(wn28)} members) {np.mean(w28_runs):.5f} {np.round(w28_runs, 5)} | '
          f'change in points {np.round((np.array(w28_runs) - np.array(wide_runs)) * 1e5, 2)}')
    d_ = [(_fauc(yv28[fold42 == k], w28_hold[7][fold42 == k]) - _fauc(yv28[fold42 == k], wide_hold[7][fold42 == k])) * 1e5 for k in range(5)]; print('split 7, change by fold of the members (points):', np.round(d_, 2))
    rng28b = np.random.default_rng(29)
    for n_ in (n_pub28, n_prv28):
        d_ = []
        for _ in range(200):
            i_ = rng28b.choice(len(yv28), n_, replace=False); d_.append((_fauc(yv28[i_], w28_hold[7][i_]) - _fauc(yv28[i_], wide_hold[7][i_])) * 1e5)
        print(f'the same change on {n_} random held-out rows: {np.mean(d_):+.2f} +- {np.std(d_):.2f} points; up in {np.mean(np.array(d_) > 0):.0%} of 200 draws')
    wstk28 = LogisticRegression(C=0.01, max_iter=3000, tol=1e-6).fit(ZW28, y); wc28 = pd.Series(wstk28.coef_[0], index=wn28); t28 = wstk28.decision_function(ZWT28)
    print('largest weights with the two seeds in:'); print(wc28.reindex(wc28.abs().sort_values(ascending=False).index).head(12).round(3).to_string())
    print('the four language models:', {k: round(float(wc28[k]), 3) for k in list(new28) + list(_tw28.values()) if k in wc28.index}, '| before:', {k: round(float(wcoef[k]), 3) for k in _tw28.values() if k in wcoef.index})
    pd.DataFrame({'id': test['id'], TARGET: rankdata(t28) / len(t28)}).to_csv(OUT / 'submission_wide_seeds.csv', index=False)
    print(f'wrote submission_wide_seeds.csv; test side, distance from our wide stack {_dist28(t28, t_w):.2f}e-4')
    del ZWT28

# 3. two checks on every member of the pool (dariushafshar's fold signature; out-of-fold against test correlation with the stack)
def _foldF28(z, f): m = np.array([z[f == k].mean() for k in range(5)]); return float(m.var(ddof=1) / (z.var() / (len(z) / 5)))
f7_28 = np.zeros(len(X), int)
for f_, (_, va_i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(X)), y)): f7_28[va_i] = f_
_cols28 = [(k, ZW[:, j], ZWT[:, j]) for j, k in enumerate(wnames)] + [(k + ' (new)', lr_(v[0]), lr_(v[1])) for k, v in new28.items()]
_ro, _rt = rankdata(wide_hold[7]), rankdata(t_w)
C28 = pd.DataFrame([[k, _fauc(yv28, o_), _foldF28(o_, fold42), _foldF28(o_, f7_28), np.corrcoef(rankdata(o_), _ro)[0, 1], np.corrcoef(rankdata(t_), _rt)[0, 1]] for k, o_, t_ in _cols28],
                   columns=['member', 'OOF AUC', 'fold F, our folds', 'fold F, other folds', 'rho OOF', 'rho test']).set_index('member')
C28['rho test - rho OOF'] = C28['rho test'] - C28['rho OOF']
_lm28 = set(_tw28) | set(_tw28.values())
def _src28(k): return 'language model' if k.replace(' (new)', '') in _lm28 else 'public pool' if k in pool else 'public bank' if k.startswith('bank ') else 'trained here'
C28['source'] = [_src28(k) for k in C28.index]
print(f"fold signature F on our folds: median {C28['fold F, our folds'].median():.0f}, smallest {C28['fold F, our folds'].min():.1f}; on folds no member was built on (split 7): median {C28['fold F, other folds'].median():.2f}, largest {C28['fold F, other folds'].max():.1f}")
print('members with the weakest fold signature:'); print(C28.sort_values('fold F, our folds').head(8).round(4).to_string())
print('test correlation minus out-of-fold correlation with the wide stack, by source:'); print(C28.groupby('source')['rho test - rho OOF'].describe()[['count', 'min', '50%', 'max']].round(4).to_string())
print('the language models:'); print(C28[C28['source'] == 'language model'].round(4).to_string())
fig, ax28 = plt.subplots(2, 2, figsize=(14, 9)); (a1, a2), (a3, a4) = ax28
if W28 is not None:
    w_ = W28.sort_values('weight in the file'); yy_ = np.arange(len(w_))
    a1.barh(yy_ + 0.2, w_['weight in the file'], height=0.4, color='#c98b2b', label='in the 0.95370 file (least squares, test rows)'); a1.barh(yy_ - 0.2, w_['weight the labels choose (same sum)'], height=0.4, color='#4a6fa5', label='chosen by the labels (logistic stacker)')
    a1.axvline(0, color='k', lw=0.8); a1.set_yticks(yy_); a1.set_yticklabels(w_.index, fontsize=7); a1.legend(fontsize=7, loc='lower right')
    a1.set_title(f'the same 16 public models, two sets of weights (the file: R2 {fit28.score(XT28, AT28):.3f})', fontsize=10)
    a2.plot(M28.index, M28['AUC on labelled rows'], 'o-', color='#4a6fa5'); a2.axhline(LAB28, color='#c98b2b', ls='--', lw=1, label='the 16 models, weights fitted on labels'); a2.legend(fontsize=8, loc='lower right')
    for w, lb in [(0.0, 'public 0.95370'), (0.1, 'goodpjw2008, their stack: public 0.95368'), (0.25, 'their stack: public 0.95362')]: a2.annotate(lb, (w, M28.loc[w, 'AUC on labelled rows']), textcoords='offset points', xytext=(8, -4), fontsize=7)
    a2.set_xlabel('share of our wide stack (rest: the 0.95370 file, replayed on out-of-fold rows)'); a2.set_ylabel('AUC on labelled rows'); a2.set_title('more of a CV-backed stack: labelled rows gain at every share', fontsize=10)
a3.scatter(C28['fold F, other folds'].clip(lower=0.01), C28['fold F, our folds'].clip(lower=0.01), s=14, c=C28['source'].map({'trained here': '#4a6fa5', 'public bank': '#7fa37f', 'public pool': '#c98b2b', 'language model': '#b0413e'}))
a3.set_xscale('log'); a3.set_yscale('log'); a3.axhline(5, color='#999', ls='--', lw=1); a3.set_xlabel('fold signature F on folds no member was built on (split 7)'); a3.set_ylabel('F on our folds (seed 42)')
a3.set_title('fold signature: was this member built on our folds?', fontsize=10)
Q28 = C28[C28['rho OOF'] > 0.9]                             # the kNN member (0.57 / 0.66) is left out of the panel so the rest can be read
for s_, c_ in [('trained here', '#4a6fa5'), ('public bank', '#7fa37f'), ('public pool', '#c98b2b'), ('language model', '#b0413e')]:
    q_ = Q28[Q28['source'] == s_]; a4.scatter(q_['rho OOF'], q_['rho test - rho OOF'], s=28 if s_ == 'language model' else 14, color=c_, label=s_, zorder=3 if s_ == 'language model' else 2)
for k in Q28.index[Q28['source'] == 'language model']: a4.annotate(k.replace('goodpjw2008 ', '').replace('masayakawamata ', '').replace('megayak ', '').replace(' target-first', '').replace(' (new)', ''), (Q28.loc[k, 'rho OOF'], Q28.loc[k, 'rho test - rho OOF']), textcoords='offset points', xytext=(6, 4), fontsize=8, color='#b0413e')
a4.axhline(0, color='#999', lw=1); a4.set_xlabel('rank correlation with the wide stack, out-of-fold rows'); a4.set_ylabel('the same on the test rows, minus the out-of-fold value')
a4.set_title('is a member closer to the stack on test than out of fold?', fontsize=10); a4.legend(fontsize=8); plt.tight_layout(); plt.show()
if new28 and all(a_ > b_ for a_, b_ in zip(w28_runs, wide_runs)) and np.mean(w28_runs) - np.mean(wide_runs) >= 0.00005:
    pd.DataFrame({'id': test['id'], TARGET: rankdata(t28) / len(t28)}).to_csv(OUT / 'submission.csv', index=False); print('the two seeds pass the bar (+0.00005 on the mean of three splits, every split up): submission.csv is the stack with the seeds')
else:
    print('the two seeds are below the bar (+0.00005 on the mean of three splits, every split up): submission.csv stays as written above')
"""),
md("""### What the board said about v28 (submitted 2026-11-13)

| file | what | public LB | AUC on labelled rows |
|---|---|---|---|
| `submission_a370_wide10.csv` | 90% Aman Atar's 0.95370 file + 10% our wide stack | **0.95367** | 0.95200 (replay) |
| `submission_a370_wide25.csv` | 75% file + 25% our wide stack | 0.95359 | 0.95217 (replay) |
| `submission_wide_seeds.csv` | our wide stack + two language-model seeds (169 members) | 0.95260 | 0.95259 (held-out, mean of 3 splits) |
| `submission.csv` | our wide stack (167 members), as v23 | 0.95260 | 0.95257 |

- Ten percent of our stack costs the 0.95370 file 3 public points (1 point = 0.00001) and adds 13 on labelled rows; 25% costs 11 and adds 31. goodpjw2008 measured 2 and 8 with their own stack: two independent stacks, the same trade.
- The board and the labelled rows rank the three files (file, 90/10, 75/25) in opposite order, in steps of 3 to 8 points. v26 measured the noise of a public difference between files this close at 3 to 9 points, so the board's order of these three is not established.
- The two seeds: +0.00000 on the board, +1.7 points held-out. The public-sized noise of that change is ± 7.6 points (v28), so the board could not have shown it.

### v29: the 0.95388 file, rebuilt byte for byte and scored on labelled rows

**The question.** On 2026-11-14 the public best is **0.95388**, [areebbb's Bold, Then Bolder](https://www.kaggle.com/code/areebbb/s6e11-bold-then-bolder-lb-0-95388), one step past
[goodpjw2008's LM Ladder | Public and CV Up](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-public-and-cv-up-lb-0-95386) (0.95386).
Both report a gain on out-of-fold rows in all five folds, which no earlier ladder did. What does that gain mean next to a stack fitted on the labels?

**How the files are made (goodpjw2008's method).**
1. Start from the 0.95370 file with a small language-model step: `base = rank(file + 0.06 (LM seed 7 - megayak's LM))`.
2. Move towards 87 public members (the 16 models of Aman Atar's bundle, goodpjw2008's wide LM stack and 70 models of goodpjw2008's OOF bank): `score = (1 - sum w) logit(base) + sum w_k logit_rank(member_k)`.
3. Pick the weights `w` to raise two things together: the out-of-fold AUC of the same mix, and a *forecast of the public score*.
   The forecast is a ridge model fitted on goodpjw2008's 44 scored files: each scored file is one linear equation about the public rows' labels, so a new file's public score can be predicted from how it re-ranks rows (1.4 points error, leave one out).
   Three settings of that trade gave three files: careful (0.95376), balanced (0.95380), bold (0.95386).
4. areebbb extrapolated: `bolder = bold + 0.5 (bold - balanced)`, public **0.95388**.

Both notebooks publish every weight, and each input file is pinned by SHA-256. This cell rebuilds the bolder file from those weights and checks its SHA-256 against areebbb's published one.

**The out-of-fold side.** The file itself has no out-of-fold predictions. goodpjw2008 replays it: least squares of the 0.95370 file on the 16 bundle models *and* their wide stack (test rows), the same weights on the out-of-fold columns, plus the same LM step. Call that the reference.
The mix weights are then applied to the 87 members' out-of-fold columns. The weights were chosen with these out-of-fold rows in view, so every mix number below is **in-sample** (optimistic); goodpjw2008's nested check keeps about 60% of it for the bold mix (+19.5 of +32.8).
Our referee is the same as in v28: the wide stacker's held-out score on split 7, where every row is scored by a stacker that did not see it.

**What we measured** (offline on the published files before this version ran, `scratch/banks/v29_1114/sim29.py`; the cell below prints its own numbers):

| | AUC on labelled rows | points against our held-out stack | public LB |
|---|---|---|---|
| the 0.95370 file, replayed (16 models) | 0.95186 (v28) | -70 | 0.95370 |
| the reference: 16 models + their wide stack + LM step | 0.95208 | -48 | 0.95370 |
| careful (in-sample) | 0.95237 | -20 | 0.95376 |
| balanced (in-sample) | 0.95227 | -29 | 0.95380 |
| bold (in-sample) | 0.95241 | -15 | 0.95386 |
| **bolder, the 0.95388 file** (in-sample) | **0.95246** | **-11** | **0.95388** |
| our wide stack (held-out) | **0.95257** | 0 | 0.95260 |

- **The rebuild is exact.** Our file has the SHA-256 areebbb publishes (`b0979694...`). Every test input is therefore the one they used.
- **The ladder closed most of the gap on labelled rows.** The 0.95370 file sat 70 points below our held-out stack; the 0.95388 file sits 11 below, and that number is in-sample. On random subsets of public-board size the gap is -9 ± 27 points (the file ahead in 37 of 100 draws), at private-board size -11 ± 13 (19 of 100).
  For the first time a public ladder file is within sampling noise of a label-fitted stack on labelled rows, measured by a second referee.
- **Where the gain comes from.** The largest moves of the bold mix: megayak's shallow trees +0.28, daoviet's GP + DVAE +0.14, megayak's LM -0.12, sadamtorres' XGBoost -0.10, masayakawamata's GPT-2 -0.07.
  v28 found the 0.95370 file holding megayak's shallow trees at -0.19 where the labels choose +0.14, and sadamtorres' XGBoost at 0.67 where the labels choose 0.16. The bold move undoes part of both: it moves the file towards the weights the labels choose.

**The same move on our stack.** If the 87-member move carries information that a label-fitted stack lacks, it should help ours too. Applied to our held-out wide stack instead of the reference, `(1 - sum w) stack + sum w_k member_k`:

| move | points on the reference (in-sample) | points on our held-out stack | folds up |
|---|---|---|---|
| careful | +28.4 | **+4.6** | 3 of 5 |
| balanced | +19.0 | +2.3 | 3 of 5 |
| bold | +32.8 | -1.3 | 1 of 5 |
| bolder (0.95388) | +37.1 | -6.3 | 1 of 5 |
| bold + 1.0 (bold - balanced) | +39.8 | -13.6 | 1 of 5 |
| bold + 2.0 (bold - balanced) | +40.1 | -34.6 | 1 of 5 |

- **The direction that raises the reference lowers our stack.** Along `bold - balanced`, the reference gains up to +40.8 points (areebbb's table reproduces to the decimal) and our stack loses up to 35, with one fold of five up.
  The move is a correction of the 0.95370 file's weights towards what labels choose. A stack that was fitted on the labels has no such error to correct, so the move only adds noise to it.
- **The careful mix gains +4.6 on our stack, below our +5 bar and with 3 of 5 folds.** It is also in-sample. `submission.csv` does not change.
- **The board peaks earlier than out-of-fold AUC** (areebbb: half a step 0.95388, 1.5 steps 0.95387, 2 steps 0.95385), and our stack peaks earlier still (at careful). Three referees, three different best step sizes; only the public one was used to choose.

**Board files from this version:** `submission_b388_wide10.csv` (90% the 0.95388 file + 10% our wide stack; on labelled rows +4.7 points over the file, 5 of 5 folds), `submission_b388_wide50.csv` (50 / 50; +15.3, 5 of 5, and +4.3 over our stack alone),
and `submission_wide_bold.csv` (our wide stack + the bold move; -1.3 points on labelled rows): if the board rewards the bold move on our stack too, the board and the labelled rows disagree about it.
The results go in the next version.

**Credit.** areebbb (the bolder step, the rebuild and its probes), goodpjw2008 (the method: moves chosen on CV and a public forecast together, the three mixes, the replay, the OOF bank and the wide LM stack), Aman Atar (the 0.95370 file and its bundle),
asteriospanoras (the noise of a public gap), and the authors of the 87 members (listed in both source notebooks). Please upvote them.
"""),
code("""# v29: the public 0.95388 file rebuilt from its published weights, replayed on labelled rows, and its move applied to our own stack
R29 = json.loads(R29_STR)                                  # weights of four public mixes; s: bundle model, b: OOF-bank model, wide: goodpjw2008's wide LM stack
_bank29 = _find('manifest.csv', 'oof-bank'); _bank29 = Path(_bank29).parent if _bank29 else None
_p29 = {k: _find(f, o) for k, (f, o) in {'wide_oof': ('wide_lm_stack_nested.npy', 'fraud-series'), 'wide_te': ('wide_lm_stack_test.npy', 'fraud-series'),
                                        'lm7': ('lm7_test_5fold.npy', 'compact-lm'), 'lm': ('megalm_test_sha3168.npy', 'compact-lm')}.items()}
_ok29 = AO28 is not None and _bank29 is not None and all(_p29.values())
if not _ok29: print('v29 skipped: inputs not mounted', {k: v is not None for k, v in _p29.items()}, 'bank', _bank29 is not None)
if _ok29:
    def _u29(v): return (rankdata(v) - 0.5) / len(v)
    def _L29(v): return logit(_u29(v))
    WO29, WT29 = _L29(np.load(_p29['wide_oof'])), _L29(np.load(_p29['wide_te']))
    _one = lambda A: np.column_stack([A, np.ones(len(A))])
    b29 = np.linalg.lstsq(_one(np.column_stack([XT28, WT29])), AT28, rcond=None)[0]       # goodpjw2008's replay: the 0.95370 file on 16 models + their wide stack, test rows
    REF29 = _one(np.column_stack([XO28, WO29])) @ b29 + 0.06 * (XO28[:, N28.index('lm7_5fold')] - XO28[:, N28.index('megayak_5f_lm')])
    U029 = _u29(AT28 + 0.06 * (_L29(np.load(_p29['lm7'])) - _L29(np.load(_p29['lm']))))
    SO29 = {r: np.zeros(len(X)) for r in R29}; ST29 = {r: 0 for r in R29}
    for k in R29['bolder']:                                # areebbb's order: the test-side sum must add up in the same order to give the same bytes
        if k.startswith('s:'): o_, t_ = _L29(Zb['oof_' + k[2:]]), _L29(Zb['test_' + k[2:]])
        elif k.startswith('b:'): o_, t_ = _L29(np.load(_bank29 / f'oof_{k[2:]}.npy')), _L29(np.load(_bank29 / f'test_{k[2:]}.npy'))
        else: o_, t_ = WO29, WT29
        for r in R29: SO29[r] = SO29[r] + R29[r][k] * o_; ST29[r] = ST29[r] + R29[r][k] * t_
    sw29 = {r: sum(R29[r].values()) for r in R29}
    b388 = (1 - sw29['bolder']) * logit(U029) + ST29['bolder']
    _f29 = OUT / 'submission_b388_rebuilt.csv'; pd.DataFrame({'id': test['id'], TARGET: _u29(b388)}).to_csv(_f29, index=False)
    h29 = hashlib.sha256(open(_f29, 'rb').read()).hexdigest()
    print('the 0.95388 file rebuilt:', 'SHA-256 matches areebbb\\'s published b0979694...' if h29 == 'b097969454128762dcd4929116d08fc9f00b3a6ebfe65fcbf7979ffaea3d729b' else f'SHA-256 differs ({h29[:16]}); the ranks may still agree')
    print(f'rank correlation of the rebuild with the 0.95370 file: {np.corrcoef(rankdata(b388), rankdata(AT28))[0, 1]:.6f}')
    _f29.unlink()                                          # not ours to submit: only the check is kept
    F29 = F28; a29 = a28
    def _mix29(w, s_, base): return (1 - w) * base + s_
    print(f'our wide stacker, held-out split 7: {a29:.6f} | the reference (16 models + their wide stack + LM step), replayed: {_fauc(yv28, REF29):.6f}')
    rows29 = []
    for r in ['careful', 'balanced', 'bold', 'bolder']:
        o_ = _mix29(sw29[r], SO29[r], REF29); o2_ = _mix29(sw29[r], SO29[r], F29)
        rows29.append([r, _fauc(yv28, o_), (_fauc(yv28, o_) - _fauc(yv28, REF29)) * 1e5, _ups28(o_, REF29), (_fauc(yv28, o_) - a29) * 1e5, (_fauc(yv28, o2_) - a29) * 1e5, _ups28(o2_, F29)])
    M29 = pd.DataFrame(rows29, columns=['mix', 'AUC (in-sample)', 'pts vs reference', 'folds up', 'pts vs our held-out stack', 'the move on our stack, pts', 'folds up on our stack']).set_index('mix')
    print('the four public mixes on labelled rows (1 point = 0.00001):'); print(M29.round({'AUC (in-sample)': 6, 'pts vs reference': 2, 'pts vs our held-out stack': 2, 'the move on our stack, pts': 2}).to_string())
    ts29 = [0.0, 0.5, 1.0, 1.5, 2.0]; S29 = []
    for t in ts29:                                         # steps along bold - balanced: linear in the weights, so linear in the sums
        w_ = sw29['bold'] + t * (sw29['bold'] - sw29['balanced']); s_ = SO29['bold'] + t * (SO29['bold'] - SO29['balanced'])
        o_, o2_ = _mix29(w_, s_, REF29), _mix29(w_, s_, F29)
        S29.append([t, (_fauc(yv28, o_) - _fauc(yv28, REF29)) * 1e5, _ups28(o_, REF29), (_fauc(yv28, o2_) - a29) * 1e5, _ups28(o2_, F29)])
    S29 = pd.DataFrame(S29, columns=['t', 'on the reference, pts', 'folds up', 'on our stack, pts', 'folds up ']).set_index('t')
    print('steps bold + t (bold - balanced):'); print(S29.round(2).to_string())
    bo29 = _mix29(sw29['bolder'], SO29['bolder'], REF29); rng29 = np.random.default_rng(29)
    for n_ in (n_pub28, n_prv28):
        d_ = []
        for _ in range(100):
            i_ = rng29.choice(len(yv28), n_, replace=False); d_.append((_fauc(yv28[i_], bo29[i_]) - _fauc(yv28[i_], F29[i_])) * 1e5)
        print(f'gap (the 0.95388 mix, in-sample - our held-out stack) on {n_} random rows: {np.mean(d_):+.1f} +- {np.std(d_):.1f} points; the file ahead in {np.mean(np.array(d_) > 0):.0%} of 100 draws')
    MX29 = pd.DataFrame([[w, _fauc(yv28, (1 - w) * bo29 + w * F29), (_fauc(yv28, (1 - w) * bo29 + w * F29) - _fauc(yv28, bo29)) * 1e5, _ups28((1 - w) * bo29 + w * F29, bo29)] for w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)],
                        columns=['share of our wide stack', 'AUC on labelled rows', 'pts vs the file', 'folds up']).set_index('share of our wide stack')
    print('the 0.95388 mix with our held-out wide stack:'); print(MX29.round({'AUC on labelled rows': 6, 'pts vs the file': 1}).to_string())
    _bw = pd.Series(R29['bold']); print('largest moves of the bold mix:', _bw.reindex(_bw.abs().sort_values(ascending=False).index).head(8).round(3).to_dict())
    T['the 0.95388 file (rebuilt)'] = b388
    for n_, v_ in [('submission_b388_wide10.csv', 0.9 * b388 + 0.1 * _lr(t_w)), ('submission_b388_wide50.csv', 0.5 * b388 + 0.5 * _lr(t_w)),
                   ('submission_wide_bold.csv', _mix29(sw29['bold'], ST29['bold'], _lr(t_w)))]:
        T[n_] = v_; pd.DataFrame({'id': test['id'], TARGET: rankdata(v_) / len(v_)}).to_csv(OUT / n_, index=False); print('wrote', n_, f'| distance from our wide stack {_dist28(v_, t_w):.1f}e-4, from the 0.95388 file {_dist28(v_, b388):.1f}e-4')
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(14, 4.6))
    lab_ = ['0.95370 file\\n(16 models)', 'reference\\n(+ wide, LM step)'] + list(M29.index); val_ = [_fauc(yv28, AO28), _fauc(yv28, REF29)] + list(M29['AUC (in-sample)'])
    pub_ = [0.95370, 0.95370, 0.95376, 0.95380, 0.95386, 0.95388]
    b1.bar(range(len(val_)), val_, color=['#c98b2b'] * 2 + ['#d9a864'] * 4); b1.axhline(a29, color='#4a6fa5', lw=2, label=f'our wide stack, held-out {a29:.5f} (public 0.95260)')
    for i_, (v_, p_) in enumerate(zip(val_, pub_)): b1.annotate(f'public {p_:.5f}', (i_, v_), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=7)
    b1.set_xticks(range(len(val_))); b1.set_xticklabels(lab_, fontsize=8); b1.set_ylim(min(val_) - 0.0002, a29 + 0.0002); b1.set_ylabel('AUC on labelled rows (mixes: in-sample)')
    b1.set_title('the public ladder on labelled rows: each rung closer to a label-fitted stack', fontsize=10); b1.legend(fontsize=8, loc='upper left')
    b2.plot(S29.index, S29['on the reference, pts'], 'o-', color='#c98b2b', label='on the replayed 0.95370 file (in-sample)'); b2.plot(S29.index, S29['on our stack, pts'], 'o-', color='#4a6fa5', label='the same move on our held-out stack')
    b2.axhline(0, color='#999', lw=1); bx = b2.twinx(); bx.plot([0.0, 0.5, 1.0, 1.5, 2.0], [0.95386, 0.95388, 0.95387, 0.95387, 0.95385], 's--', color='#b0413e', label='public LB (areebbb\\'s probes)'); bx.set_ylabel('public LB', color='#b0413e'); bx.ticklabel_format(axis='y', useOffset=False, style='plain')
    b2.set_xlabel('t in bold + t (bold - balanced)'); b2.set_ylabel('change in AUC, points (0.00001)'); b2.set_title('one direction, three referees, three best step sizes', fontsize=10)
    h1, l1 = b2.get_legend_handles_labels(); h2, l2 = bx.get_legend_handles_labels(); b2.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left'); plt.tight_layout(); plt.show()
""".replace('R29_STR', repr(R29_JSON))),
md("""### Reading the blend: three lessons that carry to other tabular competitions

**1. A good model can get zero weight.** In v5, RealMLP scored 0.94704 on its own, about as well as CatBoost (0.94721),
and the blend gave both weight 0. A blend does not reward a member for being good; it rewards it for being *right where the others are wrong*.
Look at the rank-correlation matrix above: members that correlate above ~0.99 with a stronger one add almost nothing.
The reverse also happens: in v6, TabM scored only 0.94708 alone, one of the weakest members, and the stacker gave it the third-largest weight (0.123),
because a neural net's errors differ from the trees'. So judge a new model by the blend AUC with and without it, never by its own score.

**2. Smaller trees won here.** 4-leaf LightGBM scored 0.94838, the same LightGBM with 63 leaves 0.94701, on the same features.
Deep trees model interactions between many columns at once. When the target is close to a sum of per-column effects
(the original data looks like a logistic formula), those interactions are mostly noise, and the deep model fits it.
Check this on your own data: one run with `num_leaves=4` costs minutes and tells you which regime you are in.

**3. Measure the combiner on rows it did not fit.** The hill climb's own number is computed on the same OOF predictions it picked weights on,
so it always looks a little better than it is. Here both combiners are refitted inside a second 5-fold split.
In v5 they tied (hill climb 0.94899, logistic stacker 0.94894): with members this similar, the choice of combiner does not matter much.
"""),
md("""## 4. How this notebook got here: what helped, what did not

Every version was submitted from this notebook. The public LB scores only part of the test rows, so a difference of a few 0.0001 can be noise.

| version | change | why | public LB |
|---|---|---|---|
| v2 | LightGBM + XGBoost + CatBoost + RealMLP, 5 folds, rank blend | a plain, honest baseline: four model families, out-of-fold weights | 0.94720 |
| v3 | target encoding inside each fold, decimal digits, the original data as lookups | trees split on ranges; an encoding gives them each value's fraud rate directly | 0.94841 |
| v4 | GPU, coarser rounded keys, more members, logistic stacker | more views of the same data, and a combiner the s6e8/s6e9 winners used | 0.94865 |
| v5 | 4-leaf / depth-2 trees, every column encoded per value, spline GLM | the target is close to additive, so shallow trees generalise better (lesson 2) | **0.94960** |
| v6 | more shallow views (depth 1, depth 3 CatBoost, raw inputs), TabM | v5's blend kept only shallow members; give it more of what it uses | **0.94982** |
| v8 | prediction cache (members load from the dataset), Optuna-tuned LightGBM, tutorial figures | tuning finds 2-leaf trees: same OOF as the hand-picked 4-leaf member, 62 min | 0.94981 |
| v9 | tuned XGBoost / CatBoost, count + triple-cross features, kNN, seed bags | v8's "what next" list, each judged by what it adds to the stack; the triple-cross members took the largest weights (0.29, 0.21), stacker held-out 0.94917 | 0.95004 |
| v10 | digit keys (cents, truncated integer levels, leading digits), target-encoded in fold, on 4 new members | the generator writes numbers digit by digit (goodpjw2008, ektarr); each digit tree member alone scores OOF 0.9495, above every earlier member, stacker held-out 0.95019 | **0.95095** |
| v11 | text only: history, lessons | the same predictions as v10 | (not submitted) |
| v12 | digit members without the 45 pair encodings, GLM on the encodings | no-pair members 0.94936 vs 0.94951 (the pairs are not the gap); the GLM took the largest stack weight (0.39) and stacker held-out rose to 0.95053, but the board did not move | 0.95093 |
| v13 | goodpjw2008's Part 3 / 5 recipe reproduced exactly (m-estimate encoder on every column and key, raw inputs, 3 and 4 leaves) | find out why our digit members trail the public ones by 0.0009 | members 0.95036 to 0.95052 alone (ours 0.9495); stacker held-out 0.95019 -> 0.95086, board only +0.00005: **0.95100** |
| v14 | the Part 5 member refitted with five encoder seeds, averaged | areebbb: seed-bagging the encoder lifted one member 0.95037 -> 0.95077 | member 0.95052 -> 0.95066, stacker held-out 0.95086 -> 0.95091: **0.95101** |
| v15 | three members whose test predictions come from one fit on all rows: a 5-seed depth-2 XGBoost bag, a joint GLM (one-hots of every value and digit key), a depth-2 XGBoost boosted from that GLM | v10-v14 raised the stacker 0.95019 -> 0.95091 and the board only 0.95095 -> 0.95101; four public notebooks gained on the board from a full-data refit | members 0.95054 / 0.95011 / 0.95042 alone, stacker held-out 0.95091 -> 0.95099 (largest weight: the GLM booster, 0.293): **0.95146** |
| v16 | 12 public prediction banks as stack members (lazyyeou, nilanjansaha123, yekenot; OOF + full-refit test) | offline: our 35 members CV 0.95092, + all banks +0.00020; areebbb's pooled stack and nina2025's h-blend lead the public board this way | stacker held-out 0.95099 -> 0.95113: **0.95176** |
| v17 | four more banks: goodpjw2008 Part 6, megayak, masayakawamata (GPT-2 keys stack), ektarr | offline on v16's 47 members: +0.00097 together (masayakawamata +0.00060 alone); stacker held-out 0.95113 -> 0.95212: **0.95211** |
| v18 | sameershaikh27's GLM booster trained on ten folds (encodings from 90% of the rows) as a bank member | offline on v17's 58 members: +0.00015, the only one of 11 new public banks above +0.00005 | goodpjw2008 renamed Part 6 to Part 7, so its two banks were not mounted: 57 members, stacker held-out 0.95212 -> 0.95225, board down: **0.95195** |
| v19 | v18 with goodpjw2008's two banks mounted again (same files, new kernel name) | separates v18's two changes: was the board drop the new member or the two lost ones? | goodpjw2008 renamed the kernel again (Part 8) before v19 ran: the banks were again not mounted, the same 57 members as v18; not submitted |
| v20 | the two goodpjw2008 banks found by the series prefix `s6e11-part-`, so a rename no longer drops them | the v19 question again: 59 members, as measured offline (stacker held-out 0.95223) | all 59 mounted, stacker held-out 0.95226; board below v17 again: **0.95202** (see "What the board said" in the v18 section) |
| v21 | queyrusi's digit CNN (reads the digit strings, no target encoding) as a bank member; goodpjw2008's kernel source moved to its Part 10 name | offline on v20's 59 members: +0.00010 / +0.00005 on two splits, highest rank correlation with a member 0.9795 | 60 members, stacker held-out 0.95226 -> 0.95235 (CNN weight 0.237); board down again: **0.95190** |
| v22 | the masayakawamata bank read from the author's archive dataset when the notebook output is missing; a count of expected vs loaded banks; megayak's four banks frozen in our dataset; goodpjw2008's kernel source moved to Part 11; a wide pool of 106 more public members, stacker C 0.01 | the source notebook's newest version has no output files: without this, our largest stack weight (0.736) drops out silently; the wide pool is what the 0.9528 public notebooks have and we did not: offline +0.00017 / +0.00013 on v21's 60 members | stacker held-out 0.95235 -> wide stacker 0.95251 (166 members): **0.95239**, the first gain on the board since v17 |
| v23 | megayak's target-first GPT-2 member (out-of-fold, on our folds) in the wide pool; goodpjw2008's kernel source moved to its Part 12 name | goodpjw2008 Part 12: +0.00006 nested CV, public 0.95285 -> 0.95298 on their stack; rank correlation 0.98 with our members | 0.95260 |
| v24 | text only: section 5 (what next) and the credits brought to v23's state | the same predictions as v23 | (not submitted) |
| v25 | TabICL measured as a member (null, section 4); the 0.95300 public recipe (65/35 rank-logit mix of a wide and a narrow stack) measured on our own two stackers' held-out rows, plus two extra files for the board | three public notebooks reached 0.95300 with a mix that lowers the author's own CV; does it hold on our OOF? | held-out wide 0.95257 / 0.95259 / 0.95255, 65/35 mix 0.95255 / 0.95259 / 0.95255 (null); submission.csv unchanged (v23's ranking, 0.95260); board probes: 0.65 our wide + 0.35 Part 9 0.95259, 0.65 our wide + 0.35 our narrow 0.95244 |
| v26 | the 0.95330 public file (Moran Z, Strike 9) measured on the test side; the 0.95300 recipe with our wide stack in Part 9's place as a board file; the noise of a public-board step, measured on held-out rows | the public best rose by ten public-board steps of 0.00001 to 0.00005, chosen by their public score: are they larger than the noise? | `submission.csv` unchanged (0.95260); `submission_p12_ours.csv` (0.65 Part 12 + 0.35 our wide) **0.95305**; noise of a public difference for files 33e-4 apart: 2.8e-4 |
| v27 | the public ladder (Strike 10 0.95340, LM Ladder 0.95346, Three Ladders 0.95347) replayed on held-out rows: steps picked on a public-sized subset, scored on a private-sized one; the 0.95347 file rebuilt and measured on the test side; the ladder's LM step from our own files as two board files | six public-board steps of 0.00001 to 0.00010 moved the public best; what is a step picked on the board worth on rows it was not picked on? | replay: public +2.2e-4, private -1.6e-4 over 100 draws; board: the LM step changed neither file at five decimals, `submission_p12_ours_lm12.csv` **0.95305** (without the step 0.95305), `submission_wide_lm12.csv` 0.95260 (0.95260) |
| v28 | the public 0.95370 file (Aman Atar) taken apart into 16 public models and replayed on out-of-fold rows against our held-out stack; two more language-model seeds (goodpjw2008) in the wide pool; a fold-signature and a test-against-OOF check on every pool member | the public best is now one file with no out-of-fold score: what is in it, and what does its mix score on labelled rows? | the file's mix on labelled rows 0.95186, our held-out stack 0.95257 (-70 points for the file; on the public board +110); the same 16 models with weights fitted on labels 0.95211; the two seeds +1.2 to +2.1 points on three splits, below the +5 bar, `submission.csv` unchanged (0.95260); board: `submission_a370_wide10.csv` **0.95367** (the title), `submission_a370_wide25.csv` 0.95359, `submission_wide_seeds.csv` 0.95260 |
| v29 | the public 0.95388 file (areebbb, one step past goodpjw2008's 0.95386 bold mix) rebuilt byte for byte from its published weights and replayed on labelled rows; the same 87-member move applied to our held-out stack; v28's board results; two v28 numbers corrected | the first public ladder file that reports a gain on out-of-fold rows in all five folds: is it a gain over a label-fitted stack, or over the 0.95370 file? | the 0.95388 mix on labelled rows 0.95246 in-sample, our held-out stack 0.95257 (-11 points; the 0.95370 file was -70); the move on our stack: careful +4.6 (3 of 5 folds), bold -1.3, bolder -6.3 (1 of 5); `submission.csv` unchanged (0.95260); board files `submission_b388_wide10.csv`, `submission_b388_wide50.csv`, `submission_wide_bold.csv`: see the next version |

**What did not work, or did not help enough to keep**

- **All 276 pairwise crosses.** The usual playground move. On three earlier playgrounds it measured -0.0036, -0.0004 and +0.0002;
  with 24 columns most pairs are noise and each one is a chance to overfit. Here only pairs among the ten strongest columns are encoded.
- **Deep trees as the main model.** 63-leaf LightGBM 0.94701 vs 4-leaf 0.94838 (lesson 2).
- **A model trained only on the original 20k rows** (LightGBM started from its predictions, v4/v5): 0.94459 alone and weight 0 in the blend.
  The original data is used as lookup features instead, which keeps what it knows without its smaller, slightly different distribution.
- **RealMLP**: strong alone (0.94704), weight 0 in the blend (lesson 1). It stays in the notebook as a reference for how to run it.
- **A plain logistic regression and a small extra-trees LightGBM** (v4): replaced in v5 by the spline GLM and the shallow trees, which give the blend the same kind of view (linear, low-interaction) in a stronger form.
- **kNN fraud rate** (v9): the fraud rate of each row's 100 nearest neighbours, on GPU. OOF 0.89930 alone and stacker weight -0.018: trees on encoded columns already know what a neighbourhood knows.
- **Tuning XGBoost and CatBoost with Optuna** (v9): both searches picked the shallowest option (XGBoost depth 1, CatBoost depth 3) and the tuned members took stacker weights 0.001 and 0.082. Tuning confirmed lesson 2 and added almost nothing.
- **Seed bags of the depth-1/2 XGBoost** (v9): OOF 0.94855 / 0.94844 against 0.94837 / 0.94831 single, weights 0.097 / 0.031. A small gain, not worth three times the training.
- **A better stacker that the board does not see** (v12-v14): the stacker's held-out AUC rose 0.95019 -> 0.95091 over three versions (the GLM on the encodings, the Part 3 recipe, the encoder-seed bag), and the public LB moved 0.95095 -> 0.95101.
  The v14 bag alone: member +0.00014 OOF, board +0.00001. More OOF accuracy from the same 80%-of-rows test predictions stopped paying; v15 changes the test side instead.
- **More public banks that the stack already knew** (measured offline on v17's 58 members, logistic stacker C 0.05, one 5-fold split; bar +0.00005):
  goodpjw2008's 5-seed GLM boost +0.00004, its XGBoost GLM boost +0.00002, its residual MLPs (an MLP added to the GLM logit) +0.00003 and +0.00001,
  its joint-GLM bag 0.00000, its EBM bag -0.00003; sameershaikh27's 5-fold, 3-seed GLM boost +0.00002;
  sadamtorres' TabPFN-3.5 (OOF 0.94883, a different family, rank correlation 0.983) -0.00002 and tuned CatBoost +0.00001; takayuki0127's GLM -0.00002.
  Our TabICL member (v24's kernel s6e11-tabicl-member: 4 estimators, 60k-row context, 5 folds; OOF 0.94503, rank correlation 0.975 with its nearest member)
  on v19's 59 members, splits 7 and 11: +0.00002 / -0.00004. A foundation model ranks rows differently and still adds nothing the stack lacks.
  Earlier (2026-11-03): masayakawamata's GPT-2 target-first classifier -0.00010, seven more banks between -0.00010 and +0.00002,
  and GPT-2 / GPT-4 / BERT / T5 token keys -0.00002 to 0.00000. A strong model alone is not enough: the stack has to lack what it knows.
- **The four ideas of v10's "what next" list** (measured offline for v20 on v19's 59 members, logistic stacker C 0.05, splits 7 and 11; bar +0.00005 on both).
  Alone, three of them beat the digit member they extend (0.94951): digit keys + triple crosses in one member 0.94974, digit keys of the integer and count columns 0.94960,
  learning rate 0.01 0.94959; interaction constraints (a tree splits within one source column's group) 0.94934.
  On top of the stack: +0.00001 / -0.00006, +0.00003 / +0.00001, +0.00002 / -0.00003, +0.00003 / -0.00001. Rank correlation with an existing member 0.991-0.998:
  better versions of a member the stack already has add nothing. Code: `tools/whatnext_member.py`, `tools/whatnext_addon.py` in github.com/alexchilton/kaggley (s6e11-cc-fraud).
- **queyrusi's GLM + shallow-tree blend** (measured offline for v21 on v20's 59 members, splits 7 and 11): OOF 0.95107 alone, rank correlation 0.9938 with a member already in the stack, +0.00001 / +0.00002.
  The same author's digit CNN, weaker alone (0.94974) and less correlated (0.9795), passed the bar and is in v21.
- **A better stacker that the board does not see, again (v18-v21).** Every version since v17 raised our stacker's held-out AUC and lowered the public LB:
  0.95212 / **0.95211** (v17), 0.95225 / 0.95195 (v18), 0.95226 / 0.95202 (v20), 0.95235 / 0.95190 (v21). Two other authors measured the same thing on their own stacks.
  [SIDHAARTH SHREE](https://www.kaggle.com/code/sidhaarthshree/s6e11-100-self-trained-stack-lb-0-95154) (section 6), every model self-trained: a plain 7-member stack CV 0.95084, **LB 0.95154**;
  adding GLM-boost and residual-MLP members (anything that starts from a GLM's logit) CV 0.95133-0.95136, LB 0.95141; GLM-boost alone CV 0.95128, LB 0.95097.
  [Sameer Shaikh](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-pooled-stack-of-public-oof-banks): 89 -> 93 pooled members, nested CV +0.00004, LB 0.95191 -> 0.95189.
  The public score covers 20% of the test rows and moves about ±0.0003 between random draws, so each pair is noise-sized; what is not noise-sized is that three independent stacks moved the same way.
  HYPOTHESIS (unverified): the GLM-started members, which hold large stack weights (in v21: ten-fold GLM boost 0.421, goodpjw2008 and megayak GLM boost 0.234 each, megayak GLM boost XGBoost 0.180),
  fit something in the 700k training rows that the public 20% does not reward. The private board will say which.
- **Blending with a public stack's file** (measured offline for v22, splits 7 and 11): a rank blend of 0.7 goodpjw2008's Part 11 wide stack (nested OOF 0.95250) and 0.3 our wide stack
  0.95252 / 0.95251 against our wide stack alone 0.95250 / 0.95245, +0.00002 / +0.00006. goodpjw2008's wide stack as one more member of v21's 60: 0.95241 / 0.95240, below the wide pool's 0.95250 / 0.95245.
  A blend of two stacks built from the same public members adds little out of fold; its public-board gain (+0.00018 for goodpjw2008) is the size of the public noise.
- **The 0.95300 two-stack recipe on our own stacks** (v25, held-out splits 7 / 8 / 9): wide stack alone 0.95257 / 0.95259 / 0.95255, 0.65 wide + 0.35 our 60-member stack 0.95255 / 0.95259 / 0.95255.
  On the board: 0.65 our wide + 0.35 goodpjw2008 Part 9 0.95259, 0.65 our wide + 0.35 our 60-member stack 0.95244, against 0.95260 for the wide stack alone.
  The narrow stack holds no member the wide stack lacks, so a mix can only reweight what the wide stacker already fitted; on held-out rows and on the board it did not help.
- **A ladder step from a fitted stack** (v27, all 700k held-out rows of the wide stacker, split 7): + 0.12 (megayak LM - stack) -0.51e-4, + 0.12 (masayakawamata GPT-2 - stack) -0.75e-4,
  a step of 0.12 away from the MLP bag -0.63e-4, away from the GB bag -0.55e-4. Six such steps picked on a public-sized subset: +2.2e-4 there, -1.6e-4 on the private-sized rest (100 draws, 8% positive).
  The stacker already holds each member at the weight the labelled rows support; a step picked on 60k rows moves it off that weight.
  On the board (v27, 2026-11-12) the LM step changed neither of our files at five decimals: 0.95305 with and without it, 0.95260 with and without it.
- **A second seed of each language model, as two more pool members** (v28, held-out splits 7 / 8 / 9): +2.1 / +1.2 / +1.9 points (1 point = 0.00001), every split up, below the +5 bar for the submitted stack.
  Seed 7 of megayak's LM alone +0.6, seed 2026 of masayakawamata's GPT-2 alone +1.5 (split 7). Real and small: on a public-sized subset the same change is +1.3 ± 7.6 points.
- **The best public file's weights, on labelled rows** (v28): the 0.95370 file's mix of 16 public models scores 0.95186 out of fold; the same 16 models with weights fitted on labels 0.95211; our wide stack 0.95257.
  We do not submit the file as ours. Two mixes of it with our stack are written for the board, because the board score of a mix next to its labelled-row score is the measurement.
  On the board (2026-11-13): 90% file + 10% our stack 0.95367, 75 / 25 0.95359; on labelled rows the same two mixes are +13 and +31 points above the file. The two referees rank them in opposite order.
- **The public ladder's move on our own stack** (v29, held-out split 7; the move's weights were chosen on these out-of-fold rows, so a gain would be optimistic): goodpjw2008's careful mix +4.6 points (3 of 5 folds), balanced +2.3 (3 of 5), bold -1.3 (1 of 5), areebbb's bolder -6.3 (1 of 5), two steps past bold -34.6.
  On the replayed 0.95370 file the same moves gain +19 to +41 points. The move corrects that file's weights towards what labels choose (megayak's shallow trees from -0.19 back up, sadamtorres' XGBoost down); a stack fitted on labels has no such error to correct.
- **A stronger stacker penalty alone** (v22, splits 7 and 11): C 0.01 instead of 0.05 on v21's 60 members, 0.95231 / 0.95230 against 0.95233 / 0.95232. It only helps once the pool is wide.
- **A bug worth knowing about.** The first nested hill climb summed fold weights without normalising them, so each fold's predictions
  were on a different scale and the pooled AUC read 0.895. When an honest re-score is far below the in-sample one, suspect the code before the model.
"""),
code("""
hist = pd.DataFrame({'version': ['v2', 'v3', 'v4', 'v5', 'v6', 'v8', 'v9', 'v10', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v20', 'v21', 'v22', 'v23', 'v25', 'v26', 'v27', 'v28'], 'LB': [0.94720, 0.94841, 0.94865, 0.94960, 0.94982, 0.94981, 0.95004, 0.95095, 0.95093, 0.95100, 0.95101, 0.95146, 0.95176, 0.95211, 0.95195, 0.95202, 0.95190, 0.95239, 0.95260, 0.95259, 0.95305, 0.95305, 0.95367],
                     'change': ['4-family baseline', 'in-fold target encoding', 'more members, stacker', 'shallow trees', 'more shallow views, TabM', 'cache, tuned LightGBM', 'triple crosses', 'digit keys', 'no-pair digits, GLM', 'Part 3 recipe', '5 encoder seeds', 'full refit, joint GLM', 'public banks', 'four more banks', 'ten-fold bank', 'banks re-mounted', 'digit CNN', 'wide pool', 'second LM', 'mix + Part 9 (probe)', 'Part 12 + ours (probe)', '+ LM step (probe)', '0.95370 file + ours (probe)']})
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(hist['version'], hist['LB'], 'o-', color='#4a6fa5')
for _, r in hist.iterrows(): ax.annotate(r['change'], (r['version'], r['LB']), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.set_ylabel('public LB AUC'); ax.set_title('Public LB per version of this notebook'); ax.set_ylim(0.9465, 0.9542); plt.show()
"""),
md("""## 5. What next

v10's digit keys moved the leaderboard more than any change since v5 (+0.0009). The lesson: in a synthetic playground, ask how the generator wrote the numbers, not only what they mean.

Status at v23 (public 0.95260); TabICL measured for v25, a null (section 4); the 0.95300 two-stack recipe measured for v25, a null on held-out rows and on the board (section 3, v25); the 0.95330 file and the noise of a public-board step measured for v26 (section 3, v26); the public ladder replayed on held-out rows for v27 (section 3, v27); the 0.95370 file taken apart, two more language-model seeds and the pool checks for v28 (section 3, v28); the 0.95388 file rebuilt and its move applied to our stack for v29 (section 3, v29: the ladder's move corrects the 0.95370 file towards label-fitted weights and does not help a label-fitted stack). Our own tree members are saturated: the four ideas this list held since v10 were all measured for v20, each made a slightly better member and none moved the stack.
The gains since then came from members of another family: v22's wide pool of public members (its GPT-2 member took the largest weight) and v23's target-first GPT-2 (megayak). That is the rule this list now follows: a new member must rank rows differently, below ~0.99 rank correlation with every member, or it will not add.
- **Our own target-first GPT-2 seed on our folds**: a second seed added +0.00014 on fold 0 for goodpjw2008. v28 measured goodpjw2008's two published seeds in our pool: +0.00002 on held-out rows. More seeds of the same two models will add less each time; a third kind of language model (another architecture or row format) is the open question.
- **Forensic and density features**: how the generator wrote each row, as in S6E8.
- **The test side**: full-data refits for our strongest members, as v15 did for three of them.

Companion notebook: [Four months, same screens: what held](https://www.kaggle.com/code/alexchilton/four-months-same-screens-what-held-s6e11), how to measure the noise floor an idea must beat, and how the same feature screens changed verdict over four playgrounds.

## Credits

- Competition and synthetic data: Kaggle Playground Series S6E11; original data by Udit Jain.
- RealMLP: D. Holzmüller, L. Grinsztajn, I. Steinwart, *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data* (NeurIPS 2024), pytabkit.
- TabM: Y. Gorishniy, A. Kotelnikov, A. Babenko, *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling* (ICLR 2025), via pytabkit.
- Target-first language-model members: C. Deotte's GPT-2 recipe ([kaggle-playground-solutions](https://github.com/cdeotte/kaggle-playground-solutions), Apache 2.0), as trained and published by
  [megayak](https://www.kaggle.com/datasets/megayak/s6e11-target-first-lm) and [masayakawamata](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof).
- Public members and prediction banks in the stack and the wide pool (each listed in the notebook's sources): goodpjw2008 (OOF bank, wide stack Part 12),
  queyrusi (digit CNN), sadamtorres (TabPFN and GBDT pairs), sameershaikh27 (ten-fold GLM boost, digit-sequence CNN), masayakawamata (public stack, tabpack),
  amanatar (Strike 10 file, the 0.95370 file and its bundle of 16 models), Moran Z (Strike 9 file), areebbb (the ladder recipes and their probes),
  goodpjw2008 (seed 7 of the compact LM, seed 2026 of the GPT-2, the replay of a file's weights on out-of-fold rows, the public-forecast mixes behind the 0.95386 file), areebbb (the 0.95388 bolder step), dariushafshar (the pool checks), asteriospanoras (the noise of a public gap),
  malyshevdanil (pooled stack), wguesdon (tree ensembles), lazyyeou (digit-trie target encoding), nilanjansaha123 (digit trie with GPT-2), ektarr (RealMLP stacking).
- Ideas taken from public S6E11/S6E10 notebooks, measured here: shallow trees, per-value encoding and the spline GLM from
  [sidhaarthshree](https://www.kaggle.com/code/sidhaarthshree/s6e11-fraud-per-value-te-shallow-trees-stack) and
  [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e11-part-2-two-view-tree-blend-lb-0-94945); RealMLP settings from
  [yekenot](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit) via [ravi20076](https://www.kaggle.com/code/ravi20076/playgrounds6e11-public-baseline-v1).
"""),
]


def build(cat_iters=3000, mlp_epochs=16, version=VERSION):
    nb = nbf.v4.new_notebook()
    nb.cells = [c.copy() for c in cells if mlp_epochs or 'RealMLP_TD_Classifier' not in c.source]   # 0 epochs: no RealMLP cell
    core = (Path(__file__).parent / "fp_core.py").read_text()                 # the fingerprint code, shared with tools/rekey_oof.py
    for c in nb.cells:
        c.source = c.source.replace("__FP_CORE__", core.strip())
    # the two run-length knobs live in the first code cell, where a reader can change them
    nb.cells[2] = code(nb.cells[2].source.replace("TARGET = 'is_fraud'", f"TARGET = 'is_fraud'\nCAT_ITERS, MLP_EPOCHS = {cat_iters}, {mlp_epochs}   # CatBoost max iterations, RealMLP epochs\nVERSION = '{version}'   # this notebook's version, recorded with each member it trains"))
    if not mlp_epochs:                               # the text must not promise a model the run does not train
        for c in nb.cells:
            if c.cell_type == 'markdown':
                c.source = (c.source.replace('LightGBM, XGBoost, CatBoost and RealMLP', 'LightGBM, XGBoost and CatBoost (RealMLP returns in a later version: too slow on a Kaggle CPU)')
                            .replace("                       -> RealMLP  ─┘\n", "").replace("-> CatBoost ─┤", "-> CatBoost ─┘"))
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / "s6e11-first-look.ipynb")
    meta = {"id": f"alexchilton/{SLUG}", "title": TITLE, "code_file": "s6e11-first-look.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": False, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "keywords": ["tabular", "binary classification", "ensembling", "eda"], "dataset_sources": ["uditjain13/credit-card-fraud-detection-2026", "alexchilton/s6e11-oof-predictions", "queyrusi/fraud-shared-digit-cnn-weights", "masayakawamata/s6e11-public-stack-v10-outputs", "alexchilton/s6e11-frozen-public-banks", "sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs", "masayakawamata/s6e11-gpt2-target-first-oof", "masayakawamata/s6e11-tabpack-cpu-starter-predictions", "wguesdon/s6e11-tree-ensemble-predictions", "malyshevdanil/s6e11-pooled-stack-v9", "goodpjw2008/s6e11-oof-bank", "megayak/s6e11-target-first-lm", "goodpjw2008/s6e11-fraud-series-artifacts", "moranzavdi/s6e11-strike9-lb-095330", "amanatar/s6e11-strike10-podium-apex-095340", "goodpjw2008/s6e11-compact-lm-seed7", "amanatar/s6e11-rank1-q17-meta-bundle", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"], "kernel_sources": ["lazyyeou/s6e11-digit-trie-te-joint-glm-nested-stack", "nilanjansaha123/96-key-digit-trie-gpt-2-full-refit-hybrid", "yekenot/ps-s6-e11-realmlp-pytabkit", "goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298", "megayak/s6e11-per-value-digit-te-shallow-tree-stack", "masayakawamata/s6e11-logreg-stacking-same-split-oofs", "ektarr/lb-0-95121-realmlp-stacking", "sameershaikh27/s6e11-fraud-glm-boost-on-ten-folds", "sameershaikh27/s6e11-fraud-digit-sequence-cnn-member"],
            "competition_sources": ["playground-series-s6e11"], "model_sources": []}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", OUT / "s6e11-first-look.ipynb")


if __name__ == "__main__":
    import sys
    build(*[int(x) for x in sys.argv[1:3]])
