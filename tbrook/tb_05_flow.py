
import os
import fnmatch
from matplotlib import pyplot
import scipy.io as scio
import string  as stri


def main():
    x = [2.227558088751031562e-01, 3.762581161582065903e+00, 3.575532385772475230e+00, 9.977721898714235760e-01,
1.642647217170472684e+00,
1.000000000000000021e-03,
3.845209771759799944e-01,
1.831698334916294471e+02,
3.008705128779390137e+01,
4.070848072237970294e-01,
7.500000000000000000e-01,
9.899950444061684784e-01,
3.094352916366601147e-01,
7.947036089576635653e-01,
6.076919521221795240e-01]

    x = [5.2450746e-01,   3.4911260e-01,   3.5765691e-01,   1.9428972e-01,   2.0382862e-01,
         8.3456379e-01,   7.9010386e-01,   3.7396474e-01,  6.8467550e-01,   9.9731660e-01,
         9.0232936e-01,   5.2322839e-01,   3.4284681e-01,   2.5233649e-01,   7.1459769e-01]


    measuredflow, simulatedflow = tbrook_05(x)
    sse = 0
    for i in range(len(measuredflow)):
        sse = sse + (measuredflow[i] - simulatedflow[i])*(measuredflow[i] - simulatedflow[i])
    print(sse)


def tbrook_05(x):
    # ------------------- De - normalize the  parameter values - ------------------#
    x_min = [-5, -5, 1.5, 0.01, 1, 0.001, 0.001, 0.001, 0.001, 0.01, 0.75, 0, 0, 0, 0]
    x_max = [5,  5,   8,    1, 24,   500,     1,   500,   180,    1, 1.25, 1, 1, 1, 1]
    i =14
    while i>=0:
          x[i]= x_min[i]+ x[i] *(x_max[i] - x_min[i])
          i = i - 1
    # ---------------------------------------------------------------------------------#
    # ---------------------\/---Modify the .bsn file---\/----------------------#
    bsn_new_str = x[0:5] + x[9:10]  # the new parameter values for this iteration
    bsn_linenum = [4, 5, 6, 8, 20, 13]  # the respective line numbers of these parameters in the .bsn file
    f = open("basins.bsn", "rb")
    bsn_file_copy = f.readlines()
    for i in range(len(bsn_new_str)):
        var1 = bsn_file_copy[(bsn_linenum[i] - 1)]
        str1 = '                    '  # clear the space blank2(20)for the new parameter value
        bsn_file_copy[(bsn_linenum[i] - 1)] = str1[:(16 - len(str(bsn_new_str[i])))] + str(bsn_new_str[i]) + var1[16:]
    f.close()
    f = open("basins.bsn", "wb")
    for item in bsn_file_copy:
        f.write("%s" % item)
    f.close()
    #----------------------\/---Modify the .gw files---\/-------------------
    gw_new_str = x[5:8]  # the new parameter values for this iteration
    gw_linenum = [4, 5, 6]  # the respective line numbers of these parameters in the .bsn file
    gwfilelist1 = []
    for f in os.listdir(os.curdir):
        if fnmatch.fnmatch(f, '00*.gw'):
            gwfilelist1.append(f)
    for i in range(len(gwfilelist1)):
        f = open(gwfilelist1[i], "rb")
        gw_file_copy = f.readlines()
        for j in range(len(gw_new_str)):
            var1 = gw_file_copy[(gw_linenum[j] - 1)]
            str1 = '                    '  # clear the space blank2(20)for the new parameter value
            gw_file_copy[(gw_linenum[j] - 1)] = str1[:(16 - len(str(gw_new_str[j])))] + str(gw_new_str[j]) + var1[16:]
        f.close()
        f = open(gwfilelist1[i], "wb")
        for item in gw_file_copy:
            f.write("%s" % item)
        f.close()

    # ----------------------\ / ---Modify   the.hru    files - --\ / -------------------- #
    hru_new_str = x[8:10]  # the new parameter values for this iteration
    hru_linenum = [6, 10]  # the respective line numbers of these parameters in the .hru files
    hrufilelist1 = []
    for f in os.listdir(os.curdir):
        if fnmatch.fnmatch(f, '00*.hru'):
            hrufilelist1.append(f)
    for i in range(len(hrufilelist1)):
        f = open(hrufilelist1[i], "rb")
        hru_file_copy = f.readlines()
        for j in range(len(hru_new_str)):
            var1 = hru_file_copy[(hru_linenum[j] - 1)]
            str1 = '                    '
            hru_file_copy[(hru_linenum[j] - 1)] = str1[:(16 - len(str(hru_new_str[j])))] + str(hru_new_str[j]) + var1[
                                                                                                                 16:]
        f.close()
        f = open(hrufilelist1[i], "wb")
        for item in hru_file_copy:
            f.write("%s" % item)
        f.close()
    # ----------------------\ / ---Modify    the.mgt    files - --\ / -------------------- #
    mgt_new_str = x[10:11]  # the new parameter values for this iteration
    mgt_linenum = [11]  # the respective line numbers of these parameters in the .mgt files
    cn2_base_values = [77, 77, 77, 77, 70, 70, 70, 70, 72, 72, 72, 72, 79, 79, 79, 83, 83, 83, 83]
    mgtfilelist1 = []
    for f in os.listdir(os.curdir):
        if fnmatch.fnmatch(f, '00*.mgt'):
            mgtfilelist1.append(f)
    for i in range(len(mgtfilelist1)):
        f = open(mgtfilelist1[i], "rb")
        mgt_file_copy = f.readlines()
        for j in range(len(mgt_new_str)):
            var1 = mgt_file_copy[(mgt_linenum[j] - 1)]
            str1 = '                    '
            str2 = format(min(mgt_new_str[j] * cn2_base_values[i], 99.61), '06.2f')
            mgt_file_copy[(mgt_linenum[j] - 1)] = str1[:10] + str2 + var1[16:]
        f.close()
        f = open(mgtfilelist1[i], "wb")
        for item in mgt_file_copy:
            f.write("%s" % item)
        f.close()
    #----------------------\/---Modify the .sol files---\/--------------------#

    sol_new_str = x[11:15]  # the new parameter values for this iteration
    sol_linenum = [8, 9, 10, 11]  # the respective line numbers of these parameters in the .sol files
    f = open("sol_ranges_flow.m", "r")
    sol_ranges = f.readlines()
    f.close()
    solfilelist1 = []
    for f in os.listdir(os.curdir):
        if fnmatch.fnmatch(f, '00*.sol'):
            solfilelist1.append(f)
    for i in range(len(solfilelist1)):
        f = open(solfilelist1[i], "rb")
        sol_file_copy = f.readlines()
        soiltype = sol_file_copy[1][12:17]
        if soiltype != 'NY000':
            if soiltype == 'NY026':
                line = 2
            elif soiltype == 'NY027':
                line = 3
            elif soiltype == 'NY056':
                line = 4
            elif soiltype == 'NY059':
                line = 5
            elif soiltype == 'NY099':
                line = 6
            elif soiltype == 'NY127':
                line = 7
            elif soiltype == 'NY129':
                line = 8
            elif soiltype == 'NY132':
                line = 9
            elif soiltype == 'NY133':
                line = 10
            elif soiltype == 'NY136':
                line = 11
        for j in range(len(sol_new_str)):
            var1 = sol_file_copy[(sol_linenum[j] - 1)]
            midstr = '                                                '
            var1 = var1[:27] + midstr + var1[75:]  #clear the space for the new parameter value
            newvalue1 = float(sol_ranges[line + 12 * j - 1][8:15]) + float(sol_new_str[j]) * (float(sol_ranges[line + 12 * j - 1][40:47]) - float(sol_ranges[line + 12 * j - 1][8:15]))
            str1 = format(newvalue1, '08.2f')
            newvalue2 = float(sol_ranges[line + 12 * j - 1][16:23]) + float(sol_new_str[j]) * (float(sol_ranges[line + 12 * j - 1][48:55]) - float(sol_ranges[line + 12 * j - 1][16:23]))
            str2 = format(newvalue2, '08.2f')
            newvalue3 = float(sol_ranges[line + 12 * j - 1][24:31]) + float(sol_new_str[j]) * (float(sol_ranges[line + 12 * j - 1][56:63]) - float(sol_ranges[line + 12 * j - 1][24:31]))
            str3 = format(newvalue3, '08.2f')
            newvalue4 = float(sol_ranges[line + 12 * j - 1][32:39]) + float(sol_new_str[j]) * (float(sol_ranges[line + 12 * j - 1][64:71]) - float(sol_ranges[line + 12 * j - 1][32:39]))
            str4 = format(newvalue4, '08.2f')
            sol_file_copy[(sol_linenum[j] - 1)] = var1[:31] + str1 + var1[39:43] + str2 + var1[51:55] + str3 + var1[63:67] + str4 + var1[75:]
        f.close()
        f = open(solfilelist1[i], "wb")
        for item in sol_file_copy:
            f.write("%s" % item)  #print the new .sol files
        f.close()
    #----------------------Done    modifying    all    of    the    files - ------------------- #
    os.system('SWAT_2012.exe')  #his is the executable SWAT model
    #os.system('swatbt.exe')  #his is the executable SWAT model
    #os.system('swat2005.exe')  #his is the executable SWAT model

    # ----------------------------Read mesured data----------------------------#
    loadfile = "flow_cms_oct98_sep04.mat"
    data = scio.loadmat(loadfile)
    data1 = data['flow_cms_oct98_sep04']
    measuredflow = []
    for i in range(len(data1)):
        measuredflow.append(float(data1[i][0]))
        # ----------------------Read output.rch simulated data---------------------#
    f = open("output.rch", "rb")
    rch_file_copy = f.readlines()
    REACH = []
    RCH = []
    GIS = []
    MON = []
    AREA = []
    FLOW_IN = []
    FLOW_OUT = []
    EVAP = []
    TLOSS = []
    SED_IN = []
    SED_OUT = []
    SEDCONC = []
    ORGP_OUT = []
    MINP_OUT = []
    i = 9
    while i < len(rch_file_copy):
        dataline = stri.split(rch_file_copy[i])
        REACH.append(dataline[0])
        RCH.append(dataline[1])
        GIS.append(dataline[2])
        MON.append(dataline[3])
        AREA.append(dataline[4])
        FLOW_IN.append(dataline[5])
        FLOW_OUT.append(dataline[6])
        EVAP.append(dataline[7])
        TLOSS.append(dataline[8])
        SED_IN.append(dataline[9])
        SED_OUT.append(dataline[10])
        SEDCONC.append(dataline[11])
        ORGP_OUT.append(dataline[12])
        MINP_OUT.append(dataline[13])
        i = i + 1
    simulatedflow = []
    for i in range(len(data1)):
        simulatedflow.append(float(FLOW_OUT[273+i]))
    return measuredflow, simulatedflow


if __name__ == '__main__':
    main()





