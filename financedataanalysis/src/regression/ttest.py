from scipy import stats
import numpy as np

def ttest():

    twoplayerattacky1 = []
    twoplayerattacky2 = []
    twoplayerattacky3 = []

    # twoplayerattacky1.append(0.8696)
    # twoplayerattacky1.append(0.8193)
    # twoplayerattacky1.append(0.9678)
    # twoplayerattacky1.append(0.8889)
    # twoplayerattacky1.append(0.8971)
    # twoplayerattacky1.append(0.7995)
    # twoplayerattacky1.append(0.9649)
    # twoplayerattacky1.append(0.9463)

    # twoplayerattacky2.append(0.7023)
    # twoplayerattacky2.append(0.6354)
    # twoplayerattacky2.append(0.5288)
    # twoplayerattacky2.append(0.4881)
    # twoplayerattacky2.append(0.5183)
    # twoplayerattacky2.append(0.6546)
    # twoplayerattacky2.append(0.5813)
    # twoplayerattacky2.append(0.5702)

    # twoplayerattacky3.append(0.909)
    # twoplayerattacky3.append(0.9186)
    # twoplayerattacky3.append(0.8834)
    # twoplayerattacky3.append(0.8141)
    # twoplayerattacky3.append(0.9226)
    # twoplayerattacky3.append(0.9987)
    # twoplayerattacky3.append(0.9555)
    # twoplayerattacky3.append(0.9612)

    # twoplayerattacky2.append(0.7881)
    # twoplayerattacky2.append(0.7502)
    # twoplayerattacky2.append(0.6147)
    # twoplayerattacky2.append(0.7041)
    # twoplayerattacky2.append(0.6812)
    # twoplayerattacky2.append(0.6592)
    # twoplayerattacky2.append(0.761)
    # twoplayerattacky2.append(0.8873)

    # twoplayerattacky2.append(0.6814)
    # twoplayerattacky2.append(0.7219)
    # twoplayerattacky2.append(0.631)
    # twoplayerattacky2.append(0.679)
    # twoplayerattacky2.append(0.6484)
    # twoplayerattacky2.append(0.7486)
    # twoplayerattacky2.append(0.7197)
    # twoplayerattacky2.append(0.8879)

    # twoplayerattacky2.append(0.7737)
    # twoplayerattacky2.append(0.7487)
    # twoplayerattacky2.append(0.631)
    # twoplayerattacky2.append(0.6787)
    # twoplayerattacky2.append(0.7887)
    # twoplayerattacky2.append(0.6919)
    # twoplayerattacky2.append(0.7995)
    # twoplayerattacky2.append(0.8985)

    # twoplayerattacky2.append(0.8019)
    # twoplayerattacky2.append(0.8029)
    # twoplayerattacky2.append(0.6311)
    # twoplayerattacky2.append(0.6758)
    # twoplayerattacky2.append(0.7314)
    # twoplayerattacky2.append(0.7738)
    # twoplayerattacky2.append(0.7317)
    # twoplayerattacky2.append(0.9299)

    # twoplayerattacky1.append(0.8464)
    # twoplayerattacky1.append(0.9582)
    # twoplayerattacky1.append(0.9816)
    # twoplayerattacky1.append(0.9119)
    # twoplayerattacky1.append(0.8974)
    # twoplayerattacky1.append(0.8177)
    # twoplayerattacky1.append(0.96)
    # twoplayerattacky1.append(0.9609)

    # twoplayerattacky2.append(0.692)
    # twoplayerattacky2.append(0.7094)
    # twoplayerattacky2.append(0.5953)
    # twoplayerattacky2.append(0.6395)
    # twoplayerattacky2.append(0.5575)
    # twoplayerattacky2.append(0.6181)
    # twoplayerattacky2.append(0.6172)
    # twoplayerattacky2.append(0.611)


    # twoplayerattacky3.append(0.9372)
    # twoplayerattacky3.append(0.9346)
    # twoplayerattacky3.append(0.9983)
    # twoplayerattacky3.append(0.9639)
    # twoplayerattacky3.append(0.9977)
    # twoplayerattacky3.append(0.9991)
    # twoplayerattacky3.append(0.83)
    # twoplayerattacky3.append(0.989)


    # twoplayerattacky2.append(0.6834)
    # twoplayerattacky2.append(0.763)
    # twoplayerattacky2.append(0.693)
    # twoplayerattacky2.append(0.6547)
    # twoplayerattacky2.append(0.6433)
    # twoplayerattacky2.append(0.7097)
    # twoplayerattacky2.append(0.761)
    # twoplayerattacky2.append(0.8581)

    # twoplayerattacky2.append(0.6548)
    # twoplayerattacky2.append(0.7539)
    # twoplayerattacky2.append(0.6319)
    # twoplayerattacky2.append(0.6751)
    # twoplayerattacky2.append(0.6479)
    # twoplayerattacky2.append(0.8011)
    # twoplayerattacky2.append(0.7595)
    # twoplayerattacky2.append(0.9057)
    #
    # twoplayerattacky2.append(0.7483)
    # twoplayerattacky2.append(0.8957)
    # twoplayerattacky2.append(0.8681)
    # twoplayerattacky2.append(0.6786)
    # twoplayerattacky2.append(0.8681)
    # twoplayerattacky2.append(0.651)
    # twoplayerattacky2.append(0.7057)
    # twoplayerattacky2.append(0.9514)
    #
    # twoplayerattacky2.append(0.7759)
    # twoplayerattacky2.append(0.9218)
    # twoplayerattacky2.append(0.634)
    # twoplayerattacky2.append(0.6757)
    # twoplayerattacky2.append(0.7316)
    # twoplayerattacky2.append(0.6481)
    # twoplayerattacky2.append(0.7946)
    # twoplayerattacky2.append(0.95)
    #
    #
    # l1 = twoplayerattacky1
    # l2 = twoplayerattacky2
    # l3 = twoplayerattacky3


    multiplayerattacky1 = []
    multiplayerattacky2 = []
    multiplayerattacky3 = []

    # multiplayerattacky1.append(0.8651)
    # multiplayerattacky1.append(0.8607)
    # multiplayerattacky1.append(0.9745)
    # multiplayerattacky1.append(0.8294)
    # multiplayerattacky1.append(0.8876)
    # multiplayerattacky1.append(0.8572)
    # multiplayerattacky1.append(0.9602)
    # multiplayerattacky1.append(0.925)

    # multiplayerattacky2.append(0.6316)
    # multiplayerattacky2.append(0.6027)
    # multiplayerattacky2.append(0.629)
    # multiplayerattacky2.append(0.6718)
    # multiplayerattacky2.append(0.7455)
    # multiplayerattacky2.append(0.6836)
    # multiplayerattacky2.append(0.5827)
    # multiplayerattacky2.append(0.6304)

    # multiplayerattacky3.append(0.8715)
    # multiplayerattacky3.append(0.829)
    # multiplayerattacky3.append(0.833)
    # multiplayerattacky3.append(0.8491)
    # multiplayerattacky3.append(0.9041)
    # multiplayerattacky3.append(0.9866)
    # multiplayerattacky3.append(0.8528)
    # multiplayerattacky3.append(0.936)

    # multiplayerattacky2.append(0.6748)
    # multiplayerattacky2.append(0.7461)
    # multiplayerattacky2.append(0.6034)
    # multiplayerattacky2.append(0.6825)
    # multiplayerattacky2.append(0.6698)
    # multiplayerattacky2.append(0.6824)
    # multiplayerattacky2.append(0.7444)
    # multiplayerattacky2.append(0.8649)

    # multiplayerattacky2.append(0.707)
    # multiplayerattacky2.append(0.7466)
    # multiplayerattacky2.append(0.6314)
    # multiplayerattacky2.append(0.6789)
    # multiplayerattacky2.append(0.6526)
    # multiplayerattacky2.append(0.7898)
    # multiplayerattacky2.append(0.6992)
    # multiplayerattacky2.append(0.8902)

    # multiplayerattacky2.append(0.7564)
    # multiplayerattacky2.append(0.7974)
    # multiplayerattacky2.append(0.636)
    # multiplayerattacky2.append(0.6871)
    # multiplayerattacky2.append(0.7788)
    # multiplayerattacky2.append(0.7273)
    # multiplayerattacky2.append(0.7927)
    # multiplayerattacky2.append(0.9131)

    # multiplayerattacky2.append(0.7717)
    # multiplayerattacky2.append(0.8154)
    # multiplayerattacky2.append(0.6316)
    # multiplayerattacky2.append(0.69)
    # multiplayerattacky2.append(0.7315)
    # multiplayerattacky2.append(0.7799)
    # multiplayerattacky2.append(0.7183)
    # multiplayerattacky2.append(0.91)


    multiplayerattacky1.append(0.8466)
    multiplayerattacky1.append(0.8212)
    multiplayerattacky1.append(0.9777)
    multiplayerattacky1.append(0.9096)
    multiplayerattacky1.append(0.9007)
    multiplayerattacky1.append(0.8484)
    multiplayerattacky1.append(0.9523)
    multiplayerattacky1.append(0.9449)

    # multiplayerattacky2.append(0.6826)
    # multiplayerattacky2.append(0.7123)
    # multiplayerattacky2.append(0.5938)
    # multiplayerattacky2.append(0.5251)
    # multiplayerattacky2.append(0.6306)
    # multiplayerattacky2.append(0.7176)
    # multiplayerattacky2.append(0.5457)
    # multiplayerattacky2.append(0.766)

    multiplayerattacky3.append(0.9474)
    multiplayerattacky3.append(0.9023)
    multiplayerattacky3.append(0.8679)
    multiplayerattacky3.append(0.9998)
    multiplayerattacky3.append(0.8583)
    multiplayerattacky3.append(0.9885)
    multiplayerattacky3.append(0.7697)
    multiplayerattacky3.append(0.9364)

    # multiplayerattacky2.append(0.6572)
    # multiplayerattacky2.append(0.7238)
    # multiplayerattacky2.append(0.6288)
    # multiplayerattacky2.append(0.6452)
    # multiplayerattacky2.append(0.6689)
    # multiplayerattacky2.append(0.7136)
    # multiplayerattacky2.append(0.6824)
    # multiplayerattacky2.append(0.8332)

    # multiplayerattacky2.append(0.6193)
    # multiplayerattacky2.append(0.7387)
    # multiplayerattacky2.append(0.7005)
    # multiplayerattacky2.append(0.6956)
    # multiplayerattacky2.append(0.6506)
    # multiplayerattacky2.append(0.7811)
    # multiplayerattacky2.append(0.7475)
    # multiplayerattacky2.append(0.9022)

    # multiplayerattacky2.append(0.7166)
    # multiplayerattacky2.append(0.775)
    # multiplayerattacky2.append(0.7315)
    # multiplayerattacky2.append(0.6879)
    # multiplayerattacky2.append(0.8283)
    # multiplayerattacky2.append(0.7905)
    # multiplayerattacky2.append(0.7318)
    # multiplayerattacky2.append(0.9161)

    multiplayerattacky2.append(0.7894)
    multiplayerattacky2.append(0.8153)
    multiplayerattacky2.append(0.7332)
    multiplayerattacky2.append(0.6871)
    multiplayerattacky2.append(0.7316)
    multiplayerattacky2.append(0.8317)
    multiplayerattacky2.append(0.7433)
    multiplayerattacky2.append(0.9295)


    l1 = multiplayerattacky1
    l2 = multiplayerattacky2
    l3 = multiplayerattacky3

    ttest = stats.ttest_ind(l1, l2, equal_var=True)
    print 't-statistic independent = %6.3f manipulated testing pvalue = ' % ttest[0], "{:.2e}".format(ttest[1])

    ttest = stats.ttest_ind(l1, l3, equal_var=True)
    print 't-statistic independent = %6.3f manipulated training and manipulated testing pvalue = ' % ttest[
        0], "{:.2e}".format(ttest[1])

    ttest = stats.ttest_ind(l2, l3, equal_var=True)
    print 't-statistic independent = %6.3f manipulated testing and manipulated training/testing pvalue = ' % ttest[
        0], "{:.2e}".format(ttest[1])

    a = np.array([l1, l2, l3])
    friedmantest = stats.friedmanchisquare(*(a[i, :] for i in range(a.shape[0])))
    print 'friedmantest-statistic = %6.3f pvalue = ' % friedmantest[0], "{:.2e}".format(friedmantest[1])



if __name__ == '__main__':
    ttest()
