      subroutine vibpot(rij,v,n,np)
      implicit real*8 (a-h,o-z)
      integer, intent(in) :: n,np
      double precision, intent(in) ::rij(np,3)
      double precision, intent(out) :: v(np)

c
c     pes for h2o,
c     Harry Partridge and David W. Schwenke, J. Chem. Phys.,
c     submitted Aug. 28, 1996.
c     rij(i,1)& rij(i,2) are oh distances in au
c     rij(i,3) is hoh angle in rad
c     v(i) is pes in au
c     n is number of geometries
c
c      dimension rij(np,3),v(np),c5z(245),cbasis(245),ccore(245),
      dimension c5z(245),cbasis(245),ccore(245),
     $          crest(245),idx(245,3),fmat(15,3)
       data (idx(i,1),i=1,245)/
     $ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
     $ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     $ 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     $ 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
     $ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
     $ 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
     $ 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
     $ 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     $ 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5,
     $ 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
     $ 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
     $ 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9,
     $ 9, 9, 9, 9, 9/
       data (idx(i,2),i=1,245)/
     $ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     $ 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     $ 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
     $ 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
     $ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     $ 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
     $ 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
     $ 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     $ 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4,
     $ 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
     $ 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4,
     $ 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1,
     $ 1, 1, 1, 1, 1/
       data (idx(i,3),i=1,245)/
     $ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, 1, 2, 3, 4, 5,
     $ 6, 7, 8, 9,10,11,12,13,14, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
     $12,13, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 3, 4, 5,
     $ 6, 7, 8, 9,10,11,12, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 1,
     $ 2, 3, 4, 5, 6, 7, 8, 9,10,11, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,
     $11, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 1, 2, 3, 4, 5, 6, 7, 8,
     $ 9,10, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 1, 2, 3, 4, 5, 6, 7, 8,
     $ 9,10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
     $ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2,
     $ 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6,
     $ 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
     $ 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2,
     $ 3, 4, 5, 6, 7/
       data (c5z(i),i=1,245)/
     $ 4.2278462684916D+04, 4.5859382909906D-02, 9.4804986183058D+03,
     $ 7.5485566680955D+02, 1.9865052511496D+03, 4.3768071560862D+02,
     $ 1.4466054104131D+03, 1.3591924557890D+02,-1.4299027252645D+03,
     $ 6.6966329416373D+02, 3.8065088734195D+03,-5.0582552618154D+02,
     $-3.2067534385604D+03, 6.9673382568135D+02, 1.6789085874578D+03,
     $-3.5387509130093D+03,-1.2902326455736D+04,-6.4271125232353D+03,
     $-6.9346876863641D+03,-4.9765266152649D+02,-3.4380943579627D+03,
     $ 3.9925274973255D+03,-1.2703668547457D+04,-1.5831591056092D+04,
     $ 2.9431777405339D+04, 2.5071411925779D+04,-4.8518811956397D+04,
     $-1.4430705306580D+04, 2.5844109323395D+04,-2.3371683301770D+03,
     $ 1.2333872678202D+04, 6.6525207018832D+03,-2.0884209672231D+03,
     $-6.3008463062877D+03, 4.2548148298119D+04, 2.1561445953347D+04,
     $-1.5517277060400D+05, 2.9277086555691D+04, 2.6154026873478D+05,
     $-1.3093666159230D+05,-1.6260425387088D+05, 1.2311652217133D+05,
     $-5.1764697159603D+04, 2.5287599662992D+03, 3.0114701659513D+04,
     $-2.0580084492150D+03, 3.3617940269402D+04, 1.3503379582016D+04,
     $-1.0401149481887D+05,-6.3248258344140D+04, 2.4576697811922D+05,
     $ 8.9685253338525D+04,-2.3910076031416D+05,-6.5265145723160D+04,
     $ 8.9184290973880D+04,-8.0850272976101D+03,-3.1054961140464D+04,
     $-1.3684354599285D+04, 9.3754012976495D+03,-7.4676475789329D+04,
     $-1.8122270942076D+05, 2.6987309391410D+05, 4.0582251904706D+05,
     $-4.7103517814752D+05,-3.6115503974010D+05, 3.2284775325099D+05,
     $ 1.3264691929787D+04, 1.8025253924335D+05,-1.2235925565102D+04,
     $-9.1363898120735D+03,-4.1294242946858D+04,-3.4995730900098D+04,
     $ 3.1769893347165D+05, 2.8395605362570D+05,-1.0784536354219D+06,
     $-5.9451106980882D+05, 1.5215430060937D+06, 4.5943167339298D+05,
     $-7.9957883936866D+05,-9.2432840622294D+04, 5.5825423140341D+03,
     $ 3.0673594098716D+03, 8.7439532014842D+04, 1.9113438435651D+05,
     $-3.4306742659939D+05,-3.0711488132651D+05, 6.2118702580693D+05,
     $-1.5805976377422D+04,-4.2038045404190D+05, 3.4847108834282D+05,
     $-1.3486811106770D+04, 3.1256632170871D+04, 5.3344700235019D+03,
     $ 2.6384242145376D+04, 1.2917121516510D+05,-1.3160848301195D+05,
     $-4.5853998051192D+05, 3.5760105069089D+05, 6.4570143281747D+05,
     $-3.6980075904167D+05,-3.2941029518332D+05,-3.5042507366553D+05,
     $ 2.1513919629391D+03, 6.3403845616538D+04, 6.2152822008047D+04,
     $-4.8805335375295D+05,-6.3261951398766D+05, 1.8433340786742D+06,
     $ 1.4650263449690D+06,-2.9204939728308D+06,-1.1011338105757D+06,
     $ 1.7270664922758D+06, 3.4925947462024D+05,-1.9526251371308D+04,
     $-3.2271030511683D+04,-3.7601575719875D+05, 1.8295007005531D+05,
     $ 1.5005699079799D+06,-1.2350076538617D+06,-1.8221938812193D+06,
     $ 1.5438780841786D+06,-3.2729150692367D+03, 1.0546285883943D+04,
     $-4.7118461673723D+04,-1.1458551385925D+05, 2.7704588008958D+05,
     $ 7.4145816862032D+05,-6.6864945408289D+05,-1.6992324545166D+06,
     $ 6.7487333473248D+05, 1.4361670430046D+06,-2.0837555267331D+05,
     $ 4.7678355561019D+05,-1.5194821786066D+04,-1.1987249931134D+05,
     $ 1.3007675671713D+05, 9.6641544907323D+05,-5.3379849922258D+05,
     $-2.4303858824867D+06, 1.5261649025605D+06, 2.0186755858342D+06,
     $-1.6429544469130D+06,-1.7921520714752D+04, 1.4125624734639D+04,
     $-2.5345006031695D+04, 1.7853375909076D+05,-5.4318156343922D+04,
     $-3.6889685715963D+05, 4.2449670705837D+05, 3.5020329799394D+05,
     $ 9.3825886484788D+03,-8.0012127425648D+05, 9.8554789856472D+04,
     $ 4.9210554266522D+05,-6.4038493953446D+05,-2.8398085766046D+06,
     $ 2.1390360019254D+06, 6.3452935017176D+06,-2.3677386290925D+06,
     $-3.9697874352050D+06,-1.9490691547041D+04, 4.4213579019433D+04,
     $ 1.6113884156437D+05,-7.1247665213713D+05,-1.1808376404616D+06,
     $ 3.0815171952564D+06, 1.3519809705593D+06,-3.4457898745450D+06,
     $ 2.0705775494050D+05,-4.3778169926622D+05, 8.7041260169714D+03,
     $ 1.8982512628535D+05,-2.9708215504578D+05,-8.8213012222074D+05,
     $ 8.6031109049755D+05, 1.0968800857081D+06,-1.0114716732602D+06,
     $ 1.9367263614108D+05, 2.8678295007137D+05,-9.4347729862989D+04,
     $ 4.4154039394108D+04, 5.3686756196439D+05, 1.7254041770855D+05,
     $-2.5310674462399D+06,-2.0381171865455D+06, 3.3780796258176D+06,
     $ 7.8836220768478D+05,-1.5307728782887D+05,-3.7573362053757D+05,
     $ 1.0124501604626D+06, 2.0929686545723D+06,-5.7305706586465D+06,
     $-2.6200352535413D+06, 7.1543745536691D+06,-1.9733601879064D+04,
     $ 8.5273008477607D+04, 6.1062454495045D+04,-2.2642508675984D+05,
     $ 2.4581653864150D+05,-9.0376851105383D+05,-4.4367930945690D+05,
     $ 1.5740351463593D+06, 2.4563041445249D+05,-3.4697646046367D+03,
     $-2.1391370322552D+05, 4.2358948404842D+05, 5.6270081955003D+05,
     $-8.5007851251980D+05,-6.1182429537130D+05, 5.6690751824341D+05,
     $-3.5617502919487D+05,-8.1875263381402D+02,-2.4506258140060D+05,
     $ 2.5830513731509D+05, 6.0646114465433D+05,-6.9676584616955D+05,
     $ 5.1937406389690D+05, 1.7261913546007D+05,-1.7405787307472D+04,
     $-3.8301842660567D+05, 5.4227693205154D+05, 2.5442083515211D+06,
     $-1.1837755702370D+06,-1.9381959088092D+06,-4.0642141553575D+05,
     $ 1.1840693827934D+04,-1.5334500255967D+05, 4.9098619510989D+05,
     $ 6.1688992640977D+05, 2.2351144690009D+05,-1.8550462739570D+06,
     $ 9.6815110649918D+03,-8.1526584681055D+04,-8.0810433155289D+04,
     $ 3.4520506615177D+05, 2.5509863381419D+05,-1.3331224992157D+05,
     $-4.3119301071653D+05,-5.9818343115856D+04, 1.7863692414573D+03,
     $ 8.9440694919836D+04,-2.5558967650731D+05,-2.2130423988459D+04,
     $ 4.4973674518316D+05,-2.2094939343618D+05/
       data (cbasis(i),i=1,245)/
     $ 6.9770019624764D-04,-2.4209870001642D+01, 1.8113927151562D+01,
     $ 3.5107416275981D+01,-5.4600021126735D+00,-4.8731149608386D+01,
     $ 3.6007189184766D+01, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-7.7178474355102D+01,-3.8460795013977D+01,-4.6622480912340D+01,
     $ 5.5684951167513D+01, 1.2274939911242D+02,-1.4325154752086D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00,-6.0800589055949D+00,
     $ 8.6171499453475D+01,-8.4066835441327D+01,-5.8228085624620D+01,
     $ 2.0237393793875D+02, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 3.3525582670313D+02, 7.0056962392208D+01,-4.5312502936708D+01,
     $-3.0441141194247D+02, 2.8111438108965D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-1.2983583774779D+02, 3.9781671212935D+01,
     $-6.6793945229609D+01,-1.9259805675433D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-8.2855757669957D+02,-5.7003072730941D+01,
     $-3.5604806670066D+01, 9.6277766002709D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 8.8645622149112D+02,-7.6908409772041D+01,
     $ 6.8111763314154D+01, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 2.5090493428062D+02,-2.3622141780572D+02, 5.8155647658455D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 2.8919570295095D+03,
     $-1.7871014635921D+02,-1.3515667622500D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-3.6965613754734D+03, 2.1148158286617D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00,-1.4795670139431D+03,
     $ 3.6210798138768D+02, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-5.3552886800881D+03, 3.1006384016202D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 1.6241824368764D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 4.3764909606382D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 1.0940849243716D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 3.0743267832931D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00/
       data (ccore(i),i=1,245)/
     $ 2.4332191647159D-02,-2.9749090113656D+01, 1.8638980892831D+01,
     $-6.1272361746520D+00, 2.1567487597605D+00,-1.5552044084945D+01,
     $ 8.9752150543954D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-3.5693557878741D+02,-3.0398393196894D+00,-6.5936553294576D+00,
     $ 1.6056619388911D+01, 7.8061422868204D+01,-8.6270891686359D+01,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00,-3.1688002530217D+01,
     $ 3.7586725583944D+01,-3.2725765966657D+01,-5.6458213299259D+00,
     $ 2.1502613314595D+01, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 5.2789943583277D+02,-4.2461079404962D+00,-2.4937638543122D+01,
     $-1.1963809321312D+02, 2.0240663228078D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-6.2574211352272D+02,-6.9617539465382D+00,
     $-5.9440243471241D+01, 1.4944220180218D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-1.2851139918332D+03,-6.5043516710835D+00,
     $ 4.0410829440249D+01,-6.7162452402027D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 1.0031942127832D+03, 7.6137226541944D+01,
     $-2.7279242226902D+01, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-3.3059000871075D+01, 2.4384498749480D+01,-1.4597931874215D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 1.6559579606045D+03,
     $ 1.5038996611400D+02,-7.3865347730818D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-1.9738401290808D+03,-1.4149993809415D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00,-1.2756627454888D+02,
     $ 4.1487702227579D+01, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-1.7406770966429D+03,-9.3812204399266D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-1.1890301282216D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 2.3723447727360D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-1.0279968223292D+03, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 5.7153838472603D+02, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00/
       data (crest(i),i=1,245)/
     $ 0.0000000000000D+00,-4.7430930170000D+00,-1.4422132560000D+01,
     $-1.8061146510000D+01, 7.5186735000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $-2.7962099800000D+02, 1.7616414260000D+01,-9.9741392630000D+01,
     $ 7.1402447000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00,-7.8571336480000D+01,
     $ 5.2434353250000D+01, 7.7696745000000D+01, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 1.7799123760000D+02, 1.4564532380000D+02, 2.2347226000000D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-4.3823284100000D+02,-7.2846553000000D+02,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00,-2.6752313750000D+02, 3.6170310000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00, 0.0000000000000D+00,
     $ 0.0000000000000D+00, 0.0000000000000D+00/
       data reoh,thetae,b1,roh,alphaoh,deoh,phh1,phh2/0.958649d0,
     $      104.3475d0,2.0d0,0.9519607159623009d0,2.587949757553683d0,
     $      42290.92019288289d0,16.94879431193463d0,12.66426998162947d0/
       data f5z,fbasis,fcore,frest/0.99967788500000d0,
     $      0.15860145369897d0,-1.6351695982132d0,1d0/
      save
      data ifirst/0/
      if(ifirst.eq.0)then
       ifirst=1
c       write(6,1)
    1  format(/1x,'pes for h2o',
     $        /1x,'by Harry Partridge and David W. Schwenke',
     $        /1x,'submitted to J. Chem. Phys. Aug. 28, 1996')
c       write(6,56)
   56  format(/1x,'parameters before adjustment')
c       write(6,55)phh1,phh2,deoh,alphaoh,roh
   55  format(/1x,'two body potential parameters:',
     $        /1x,'hh: phh1 = ',f10.1,' phh2 = ',f5.2,
     $        /1x,'oh: deoh = ',f10.1,' alpha = ',f7.4,
     $        ' re = ',f7.4)
c       write(6,4)reoh,thetae,b1
    4  format(/1x,'three body parameters:',
     $        /1x,'reoh = ',f10.4,' thetae = ',f10.4,
     $        /1x,'betaoh = ',f10.4,
     $        /1x,'    i    j    k',7x,'c5z',9x,'cbasis',10x,'ccore',
     $        10x,'crest')
       do 2 i=1,245
c        write(6,5)(idx(i,j)-1,j=1,3),c5z(i),cbasis(i),ccore(i),crest(i)
    5   format(1x,3i5,1p4e15.7)
        c5z(i)=f5z*c5z(i)+fbasis*cbasis(i)+fcore*ccore(i)
     $       +frest*crest(i)
    2  continue
c       write(6,57)f5z,fbasis,fcore,frest
   57  format(/1x,'adjusting parameters using scale factors ',
     $        /1x,'f5z =    ',f11.8,
     $        /1x,'fbasis = ',f11.8,
     $        /1x,'fcore =  ',f11.8,
     $        /1x,'frest =  ',f11.8)
       phh1=phh1*f5z
       deoh=deoh*f5z
c       write(6,55)phh1,phh2,deoh,alphaoh,roh
c       write(6,58)reoh,thetae,b1,((idx(i,j)-1,j=1,3),c5z(i),i=1,245)
   58  format(/1x,'three body parameters:',
     $        /1x,'reoh = ',f10.4,' thetae = ',f10.4,
     $        /1x,'betaoh = ',f10.4,
     $        /1x,'    i    j    k   cijk',
     $        /(1x,3i5,1pe15.7))
c
c     convert parameters from 1/cm, angstrom to a.u.
c
       reoh=reoh/0.529177249d0
       b1=b1*0.529177249d0*0.529177249d0
       do 3 i=1,245
        c5z(i)=c5z(i)*4.556335d-6 
    3  continue
       rad=acos(-1d0)/1.8d2
       ce=cos(thetae*rad)
       phh1=phh1*exp(phh2)
       phh1=phh1*4.556335d-6
       phh2=phh2*0.529177249d0
       deoh=deoh*4.556335d-6
       roh=roh/0.529177249d0
       alphaoh=alphaoh*0.529177249d0
       c5z(1)=c5z(1)*2d0
      end if
      do 6 i=1,n
       x1=(rij(i,1)-reoh)/reoh
       x2=(rij(i,2)-reoh)/reoh
       x3=cos(rij(i,3))-ce
       rhh=sqrt(rij(i,1)**2+rij(i,2)**2
     $      -2d0*rij(i,1)*rij(i,2)*cos(rij(i,3)))
       vhh=phh1*exp(-phh2*rhh)
       ex=exp(-alphaoh*(rij(i,1)-roh))
       voh1=deoh*ex*(ex-2d0)
       ex=exp(-alphaoh*(rij(i,2)-roh))
       voh2=deoh*ex*(ex-2d0)
       fmat(1,1)=1d0
       fmat(1,2)=1d0
       fmat(1,3)=1d0
       do 10 j=2,15
        fmat(j,1)=fmat(j-1,1)*x1
        fmat(j,2)=fmat(j-1,2)*x2
        fmat(j,3)=fmat(j-1,3)*x3
   10  continue
       v(i)=0d0
       do 12 j=2,245
        term=c5z(j)*(fmat(idx(j,1),1)*fmat(idx(j,2),2)
     $                    +fmat(idx(j,2),1)*fmat(idx(j,1),2))
     $                    *fmat(idx(j,3),3)
        v(i)=v(i)+term
   12  continue
       v(i)=v(i)*exp(-b1*((rij(i,1)-reoh)**2+(rij(i,2)-reoh)**2))
     $      +c5z(1)
     $      +voh1+voh2+vhh
    6 continue
      return
      end

