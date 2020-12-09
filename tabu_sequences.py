from enum import Enum

class tabu_sequences():
    """sequences used in model building"""

    PEP_SEQ = ["AAKRRH","AARRH","ADFVETPTANI","ADSRKPI","AGKRKSG","AGVKDGKLDF","AILGDTAWDFGSLGGVFTSIGKA","AILGDTAWDFGSLGGVFTSIGKALHQVF","AILGDTAWDFGSLGGVFTSIGKALHQVFGAIYGAA","AKKRH","AKNARRH","AKNENSFAIAA","AKRAH","AKRKH","AKRR","AKRRH","ANAKRRH","ANDDTFALAA","ANDENYALAA","ANDETYAVAA","ANSDSFDYALAA","ARRRRHLCAA","ARRRRHLCCA","ARRRRHLCWA","ARRRRHWCAA","ARRRRHWCCA","ARRRRHWCWA","ARRRRMVCGA","ARRRRMYCCA","ARRRRWWA","ARRRRWWCGA","ARRRRWYA","ARRRRWYCGA","ARRRRYVCCA","ARRRRYWCGA","ARRWWCCA","ARRWYCGA","ASVEAPSING","AWDFGSIGGVFTSVGKLVHQVFGTAYGVL","AWDFGSLGGVFTSIGKALHQVF","AWDFGSLGGVFTSIGKALHQVFGAIYGAA","AWDFGSVGGLFTSLGKAVHQVFGSVYTIM","AWDFGSVGGLFTSLGKAVHQVFGSVYTTM","AWDFGSVGGVLNSLGKMVHQIFGSAYTAL","AWYCGA","CAGKRKSG","CDMETDNGGWTL","CGKRKLC","CGKRKSC","CGKRRC","CGYGLC","CGYKGC","CGYKLC","CGYKRC","CKRKC","DHFVKLVS","DINGGA","DLKKMNTEGQ","DTRACDVIALLCHLNT","EF","EFKEAFNMIDQTRDGFVC","EKAKKAITD","ETDNGGWTL","ETTVTGGSAGYAVSGLAGFFTPGAKQN","EVKKQR","EYFIGVN","FAAGRK","FAAGRRASL","FAAGRRCHK","FALSHMLLQHVLFLCV","FITTVVA","FLFLNAATCYRDRV","FPFDFHHDRYYHFHWKRYQH","FWFTLIKTQAKQPARYRRFC","GGARDAGKAEWW","GICRCICGRGICRCICGRIGGRVPGVGVPGVGHHHHHH","GKQNNLSLAA","GKR","GPVYLSSLFLQQFDWYLYKAEPA","GWVKPAKLDG","IGVRPGKLDL","IKTSGETLSIVPLGATPISPDSIEADILISPPDST","IRKQGISGPL","IWWRPRDWPTFIFYFREWRW","KEKRREWEWRFRWEFRLYFE","KEYFRRFFHCHNHQREWHWH","KKR","KR","KRRH","LAPSFVNEDGVE","LRQVVEVS","MAAALITSQGFSGEVTDMFLMMACLSLFETDERMSLFLSGCLS","MADPSFDLFFTKSHPIYN","MAGAFIAGLA","MAILGDTAWDFGSLGGVFTSIGKALHQVFGAIY","MAKEMSMVERNARGKQARQYFIECAGKEG","MALKLAEE","MAMELFR","MARNLLNVMS","MAVADPPPNAKAQAFVTSEAFGAALTFG","MCDLLWWIGLGLLAAGMLAFDLGPYLVKRFKKRTQDGDARYREED","MCFVLQGENQEHILIQNEYIYINVIICR","MCINFTDCKKICIFARLHTIGDTIVLSIEKQ","MCPPGQCLQSAQSPWAPTSKKRRATSRGGRQWGGSLPCACTAPQA","MCTCSNCLLSAL","MDFFLFDCCAGKLLKMKTLHYIYLADAFIQSDVQKCIS","MEISMFTTSLVFSPPSLTILVN","MELQRYWGALTTSSFRTDWTFSTPPAMA","MESLGSYRVDEVSHNRSSLMLPVEKWRDVQRGP","MESRESWDFLVWMVLSGGVILFVAYLILGAVAYYLDKKETKPSNQ","MFGETSDLQAEKRGNFVVVGCVDGPKVRIMVFKCSQ","MFSTSAKI","MGDRPDLGEIDKFDKTKLKKTETQEKNTLPTKETIEQEKKN","MGKAKFERSKPHVNIGTIGHVDHGKTTLTAA","MIEVFLFGIVLGLIPITLAGLFVTAYLQYRRGDQLDL","MIGLDPIAVLPALLAMLLLVAIRQDRVFFDLVLLR","MIVFDIVAAVLAVASVGYLVLALVKPERF","MKDTE","MKIIRKKKTTRNRQIHLNFS","MKNASSEIGLGTKKKREKNRGLTLKKNILCQFRIGLKFASSKN","MKQKLLQATD","MKTEFFSGSSIYFVLYQSNK","MKVRASVKKICRNCKVIKRNGVVRVICTEPKHKQRQG","MLGNMHVFMAVLGTILFFGFLAAYLSHKWDD","MLIGWAIEGY","MLKVCCSSPVAIGQPPLMLITRQIYKKTSTLTGTQITSAFIFLLK","MLTALFPSLMVTVCILNDFV","MLTIYILIYYIC","MMRLVIILIVLLLLSFPAY","MMTRWLFSTN","MNKKDKNLSYKDLLYVIVVVAVGTFMILKVPELFR","MNRIDKKFKELKRNDKKAFIAFIMAGDPSLSVTEK","MPDVYYEDDTDLSLLEGKTVAVIGYGSQG","MRCKGFLFDLDGTLVDSLPAVERAWSNWA","MRMAAYCRVSTDREAQLESLENQKRFFEDFAEKQGHQLVKIYAD","MRVLGWGQNEYAYQYISQATLSNKRFPASLHCGLMLVV","MSECENVSGRVNIAIVSDTALIRQYIEQQNRPH","MSLLVKLQFI","MTSLLGD","MTVKLGCPQSSQKKSHPSLF","MVAPVTTVFNAKDPSEIKKKIVRGIVRHYKVPKSK","MVEAFIRLINIKIFYIINTKYPFYKCIWPCYHISIIVLNL","MVPGLSPVTAIRTFSLTHLFGVSWIMLLIVFSSGPAAGHR","MVRSKNTGPSYSNNLGNTG","MWFRFFMIGFFSLTAISLMGYQVSEIYQAYSDMFFNKN","MWGVTTSRH","MYVGEIA","NAAKRRH","NKRRSRSSRS","NMLKRARNRV","NRPESTQCSLGGTGRKVSVTSQSGKVISSWESYKSGGETRL","NSSQDPDFTQ","PCRARIYGGCA","PLGFQIDDAKLKRAGLDYWP","PVSWQGPSLDPANPGVATEVLEITHHGFTD","PWLKPGDLDL","QFGPVFTWLNHA","QSSQGSTDKT","RGGRLCYCRRRFCVCVGR","RHWCW","RHWEQFYFRRRERKFWLFFW","RNAHNFPLDLAAIEVPSING","RQMRAWGQDYQHGGMGYSC","RRH","RRHLCW","RRHWCW","RRRRHLCW","RRRRHWCW","RTSKKR","RWMVWRHWFHRLRLPYNPGKNKQNQQWP","RWWCC","SFVNLWTPRYSL","SKKR","SSVDKDLA","SVALVPHVGMGLETRTETWMSSEGAWKHVQRIETWILRHPG","SVKYLEFIS","TGRSHCHTCKTDARTTMEATFTHVLL","THKIVADK","TTSTRR","TTSTRRASL","VCVCVCEVVV","VKSSLCTK","VTTASRQPTGTSSSALRCSPWPSGRSPSGQTALQ","WHWAWYSPTARM","WHWRLWDVPDNP","WYCW"]
