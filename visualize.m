nok = readmatrix('1_rms_125_1_Nok.csv');%'NumHeaderLines',22);
ok_1 = readmatrix('1_rms_125_1_ok.csv');  %'NumHeaderLines',22);
ok_2 = readmatrix('1_rms_125_2_ok.csv');  %'NumHeaderLines',22);


ptCloud_nok = pointCloud([nok(:,5),nok(:,6),nok(:,7)]);
ptCloud_ok_1 = pointCloud([ok_1(:,5),ok_1(:,6),ok_1(:,7)]);
ptCloud_ok_2 = pointCloud([ok_2(:,5),ok_2(:,6),ok_2(:,7)]);

figure(1);
%hold off; 
pcshow(ptCloud);
%plot(nok(:,5),nok(:,6),nok(:,7));
title('nok');

figure(2);

pcshow(ptCloud_ok_1);

title('ok_1');

figure(3);

pcshow(ptCloud_ok_2);

title('ok_2');