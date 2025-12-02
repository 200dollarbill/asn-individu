import numpy as np

class DWTCoeff:
    def __init__(self):
        pass

    def _dirac(self, k):
        return 1 if k == 0 else 0

    def get_filter(self, scale):
        j = scale
        a = -(round(2**j) + round(2**(j-1)) - 2)
        b = -(1 - round(2**(j-1))) + 1

        num_coeffs = b - a
        qj = np.zeros(num_coeffs)
        for k in range(a, b):
            coeff_val = 0
            if scale == 1:
                coeff_val = -2 * (self._dirac(k) - self._dirac(k + 1))
            
            elif scale == 2:
                coeff_val = (-1/4) * (self._dirac(k - 1) + 3 * self._dirac(k) +
                                      2 * self._dirac(k + 1) - 2 * self._dirac(k + 2) -
                                      3 * self._dirac(k + 3) - self._dirac(k + 4))
            
            elif scale == 3:
                coeff_val = (-1/32) * (self._dirac(k - 3) + 3 * self._dirac(k - 2) + 6 * self._dirac(k - 1) +
                                       10 * self._dirac(k) + 11 * self._dirac(k + 1) + 9 * self._dirac(k + 2) +
                                       4 * self._dirac(k + 3) - 4 * self._dirac(k + 4) - 9 * self._dirac(k + 5) -
                                       11 * self._dirac(k + 6) - 10 * self._dirac(k + 7) - 6 * self._dirac(k + 8) -
                                       3 * self._dirac(k + 9) - self._dirac(k + 10))
            elif scale == 4:
                coeff_val = (-1/256 ) * (self._dirac(k-7) + 3*self._dirac(k-6) + 6*self._dirac(k-5) + 10*self._dirac(k-4)
                        + 15*self._dirac(k-3) + 21*self._dirac(k-2) + 28*self._dirac(k-1) + 36*self._dirac(k)
                        + 41*self._dirac(k+1) + 43*self._dirac(k+2) + 42*self._dirac(k+3) + 38*self._dirac(k+4)
                        + 31*self._dirac(k+5) + 21*self._dirac(k+6) + 8*self._dirac(k+7) - 8*self._dirac(k+8)
                        - 21*self._dirac(k+9) - 31*self._dirac(k+10) - 38*self._dirac(k+11) - 42*self._dirac(k+12)
                        - 43*self._dirac(k+13) - 41*self._dirac(k+14) - 36*self._dirac(k+15) - 28*self._dirac(k+16)
                        - 21*self._dirac(k+17) - 15*self._dirac(k+18) - 10*self._dirac(k+19) - 6*self._dirac(k+20)
                        - 3*self._dirac(k+21) - self._dirac(k+22))
            elif scale == 5:
                coeff_val = (-1/512)*(self._dirac(k-15) + 3*self._dirac(k-14) + 6*self._dirac(k-13) + 10*self._dirac(k-12) + 15*self._dirac(k-11) + 21*self._dirac(k-10)
                        + 28*self._dirac(k-9) + 36*self._dirac(k-8) + 45*self._dirac(k-7) + 55*self._dirac(k-6) + 66*self._dirac(k-5) + 78*self._dirac(k-4)
                        + 91*self._dirac(k-3) + 105*self._dirac(k-2)+ 120*self._dirac(k-1) + 136*self._dirac(k) + 149*self._dirac(k+1) + 159*self._dirac(k+2)
                        + 166*self._dirac(k+3)+ 170*self._dirac(k+4)+ 171*self._dirac(k+5) + 169*self._dirac(k+6) + 164*self._dirac(k+7) + 156*self._dirac(k+8)
                        + 145*self._dirac(k+9) + 131*self._dirac(k+10) + 114*self._dirac(k+11) + 94*self._dirac(k+12) + 71*self._dirac(k+13) + 45*self._dirac(k+14)
                        + 16*self._dirac(k+15) - 16*self._dirac(k+16) - 45*self._dirac(k+17) - 71*self._dirac(k+18) - 94*self._dirac(k+19) - 114*self._dirac(k+20)
                        - 131*self._dirac(k+21) - 145*self._dirac(k+22) - 156*self._dirac(k+23) - 164*self._dirac(k+24) - 169*self._dirac(k+25)
                        - 171*self._dirac(k+26) - 170*self._dirac(k+27) - 166*self._dirac(k+28) - 159*self._dirac(k+29) - 149*self._dirac(k+30)
                        - 136*self._dirac(k+31) - 120*self._dirac(k+32) - 105*self._dirac(k+33) - 91*self._dirac(k+34) - 78*self._dirac(k+35)
                        - 66*self._dirac(k+36) - 55*self._dirac(k+37) - 45*self._dirac(k+38) - 36*self._dirac(k+39) - 28*self._dirac(k+40)
                        - 21*self._dirac(k+41) - 15*self._dirac(k+42) - 10*self._dirac(k+43) - 6*self._dirac(k+44) - 3*self._dirac(k+45) - self._dirac(k+46))
            elif scale == 6:
                coeff_val = (-1 / 2048 * (self._dirac(k - 31) + 3 * self._dirac(k - 30) + 6 * self._dirac(k - 29) + 10 * self._dirac(k - 28) +
                15 * self._dirac(k - 27) + 21 * self._dirac(k - 26) + 28 * self._dirac(k - 25) + 36 * self._dirac(k - 24) +
                45 * self._dirac(k - 23) + 55 * self._dirac(k - 22) + 66 * self._dirac(k - 21) + 78 * self._dirac(k - 20) +
                91 * self._dirac(k - 19) + 105 * self._dirac(k - 18) + 120 * self._dirac(k - 17) + 136 * self._dirac(k - 16) +
                153 * self._dirac(k - 15) + 171 * self._dirac(k - 14) + 190 * self._dirac(k - 13) + 210 * self._dirac(k - 12) +
                231 * self._dirac(k - 11) + 253 * self._dirac(k - 10) + 276 * self._dirac(k - 9) + 300 * self._dirac(k - 8) +
                325 * self._dirac(k - 7) + 351 * self._dirac(k - 6) + 378 * self._dirac(k - 5) + 406 * self._dirac(k - 4) +
                435 * self._dirac(k - 3) + 465 * self._dirac(k - 2) + 496 * self._dirac(k - 1) + 528 * self._dirac(k) +
                557 * self._dirac(k + 1) + 583 * self._dirac(k + 2) + 606 * self._dirac(k + 3) + 626 * self._dirac(k + 4) +
                643 * self._dirac(k + 5) + 657 * self._dirac(k + 6) + 668 * self._dirac(k + 7) + 676 * self._dirac(k + 8) +
                681 * self._dirac(k + 9) + 683 * self._dirac(k + 10) + 682 * self._dirac(k + 11) + 678 * self._dirac(k + 12) +
                671 * self._dirac(k + 13) + 661 * self._dirac(k + 14) + 648 * self._dirac(k + 15) + 632 * self._dirac(k + 16) +
                613 * self._dirac(k + 17) + 591 * self._dirac(k + 18) + 566 * self._dirac(k + 19) + 538 * self._dirac(k + 20) +
                507 * self._dirac(k + 21) + 473 * self._dirac(k + 22) + 436 * self._dirac(k + 23) + 396 * self._dirac(k + 24) +
                353 * self._dirac(k + 25) + 307 * self._dirac(k + 26) + 258 * self._dirac(k + 27) + 206 * self._dirac(k + 28) +
                151 * self._dirac(k + 29) + 93 * self._dirac(k + 30) + 32 * self._dirac(k + 31) - 32 * self._dirac(k + 32) -
                93 * self._dirac(k + 33) - 151 * self._dirac(k + 34) - 206 * self._dirac(k + 35) - 258 * self._dirac(k + 36) -
                307 * self._dirac(k + 37) - 353 * self._dirac(k + 38) - 396 * self._dirac(k + 39) - 436 * self._dirac(k + 40) -
                473 * self._dirac(k + 41) - 507 * self._dirac(k + 42) - 538 * self._dirac(k + 43) - 566 * self._dirac(k + 44) -
                591 * self._dirac(k + 45) - 613 * self._dirac(k + 46) - 632 * self._dirac(k + 47) - 648 * self._dirac(k + 48) -
                661 * self._dirac(k + 49) - 671 * self._dirac(k + 50) - 678 * self._dirac(k + 51) - 682 * self._dirac(k + 52) -
                683 * self._dirac(k + 53) - 681 * self._dirac(k + 54) - 676 * self._dirac(k + 55) - 668 * self._dirac(k + 56) -
                657 * self._dirac(k + 57) - 643 * self._dirac(k + 58) - 626 * self._dirac(k + 59) - 606 * self._dirac(k + 60) -
                583 * self._dirac(k + 61) - 557 * self._dirac(k + 62) - 528 * self._dirac(k + 63) - 496 * self._dirac(k + 64) -
                465 * self._dirac(k + 65) - 435 * self._dirac(k + 66) - 406 * self._dirac(k + 67) - 378 * self._dirac(k + 68) -
                351 * self._dirac(k + 69) - 325 * self._dirac(k + 70) - 300 * self._dirac(k + 71) - 276 * self._dirac(k + 72) -
                253 * self._dirac(k + 73) - 231 * self._dirac(k + 74) - 210 * self._dirac(k + 75) - 190 * self._dirac(k + 76) -
                171 * self._dirac(k + 77) - 153 * self._dirac(k + 78) - 136 * self._dirac(k + 79) - 120 * self._dirac(k + 80) -
                105 * self._dirac(k + 81) - 91 * self._dirac(k + 82) - 78 * self._dirac(k + 83) - 66 * self._dirac(k + 84) -
                55 * self._dirac(k + 85) - 45 * self._dirac(k + 86) - 36 * self._dirac(k + 87) - 28 * self._dirac(k + 88) -
                21 * self._dirac(k + 89) - 15 * self._dirac(k + 90) - 10 * self._dirac(k + 91) - 6 * self._dirac(k + 92) -
                3 * self._dirac(k + 93) - self._dirac(k + 94)))
            elif scale == 7:

                s7p1 = ((self._dirac(k-63) + 3*self._dirac(k-62) + 6*self._dirac(k-61) + 10*self._dirac(k-60) + 15*self._dirac(k-59) +
                        21*self._dirac(k-58) + 28*self._dirac(k-57) + 36*self._dirac(k-56) + 45*self._dirac(k-55) +
                        55*self._dirac(k-54) + 66*self._dirac(k-53) + 78*self._dirac(k-52) + 91*self._dirac(k-51) +
                        105*self._dirac(k-50) + 120*self._dirac(k-49) + 136*self._dirac(k-48) + 153*self._dirac(k-47) +
                        171*self._dirac(k-46) + 190*self._dirac(k-45) + 210*self._dirac(k-44) + 231*self._dirac(k-43) +
                        253*self._dirac(k-42) + 276*self._dirac(k-41) + 300*self._dirac(k-40) + 325*self._dirac(k-39) +
                        351*self._dirac(k-38) + 378*self._dirac(k-37) + 406*self._dirac(k-36) + 435*self._dirac(k-35) +
                        465*self._dirac(k-34) + 496*self._dirac(k-33) + 528*self._dirac(k-32) + 561*self._dirac(k-31) +
                        595*self._dirac(k-30) + 630*self._dirac(k-29) + 666*self._dirac(k-28) + 703*self._dirac(k-27) +
                        741*self._dirac(k-26) + 780*self._dirac(k-25) + 820*self._dirac(k-24) + 861*self._dirac(k-23) +
                        903*self._dirac(k-22) + 946*self._dirac(k-21) + 990*self._dirac(k-20) + 1035*self._dirac(k-19) +
                        1081*self._dirac(k-18) + 1128*self._dirac(k-17) + 1176*self._dirac(k-16) + 1225*self._dirac(k-15) +
                        1275*self._dirac(k-14) + 1326*self._dirac(k-13) + 1378*self._dirac(k-12) + 1431*self._dirac(k-11) +
                        1485*self._dirac(k-10) + 1540*self._dirac(k-9) + 1596*self._dirac(k-8) + 1653*self._dirac(k-7) +
                        1711*self._dirac(k-6) + 1770*self._dirac(k-5) + 1830*self._dirac(k-4) + 1891*self._dirac(k-3) +
                        1953*self._dirac(k-2) + 2016*self._dirac(k-1) + 2080*self._dirac(k) + 2141*self._dirac(k+1) +
                        2199*self._dirac(k+2) + 2254*self._dirac(k+3) + 2306*self._dirac(k+4) + 2355*self._dirac(k+5) +
                        2401*self._dirac(k+6) + 2444*self._dirac(k+7) + 2484*self._dirac(k+8) + 2521*self._dirac(k+9) +
                        2555*self._dirac(k+10) + 2586*self._dirac(k+11) + 2614*self._dirac(k+12) + 2639*self._dirac(k+13) +
                        2661*self._dirac(k+14) + 2680*self._dirac(k+15) + 2696*self._dirac(k+16) + 2709*self._dirac(k+17) +
                        2719*self._dirac(k+18) + 2726*self._dirac(k+19) + 2730*self._dirac(k+20) + 2731*self._dirac(k+21) +
                        2729*self._dirac(k+22) + 2724*self._dirac(k+23) + 2716*self._dirac(k+24) + 2705*self._dirac(k+25) +
                        2691*self._dirac(k+26) + 2674*self._dirac(k+27) + 2654*self._dirac(k+28) + 2631*self._dirac(k+29) +
                        2605*self._dirac(k+30) + 2576*self._dirac(k+31) + 2544*self._dirac(k+32) + 2509*self._dirac(k+33) +
                        2471*self._dirac(k+34) + 2430*self._dirac(k+35) + 2386*self._dirac(k+36) + 2339*self._dirac(k+37) +
                        2289*self._dirac(k+38) + 2236*self._dirac(k+39) + 2180*self._dirac(k+40) + 2121*self._dirac(k+41) +
                        2059*self._dirac(k+42) + 1994*self._dirac(k+43) + 1926*self._dirac(k+44) + 1855*self._dirac(k+45) +
                        1781*self._dirac(k+46) + 1704*self._dirac(k+47) + 1624*self._dirac(k+48) + 1541*self._dirac(k+49) +
                        1455*self._dirac(k+50) + 1366*self._dirac(k+51) + 1274*self._dirac(k+52) + 1179*self._dirac(k+53) +
                        1081*self._dirac(k+54) + 980*self._dirac(k+55) + 876*self._dirac(k+56) + 769*self._dirac(k+57) +
                        659*self._dirac(k+58) + 546*self._dirac(k+59) + 430*self._dirac(k+60) + 311*self._dirac(k+61) +
                        189*self._dirac(k+62) + 64*self._dirac(k+63) ))


                s7p2 = (- 64*self._dirac(k+64) - 189*self._dirac(k+65) -
                        311*self._dirac(k+66) - 430*self._dirac(k+67) - 546*self._dirac(k+68) - 659*self._dirac(k+69) -
                        769*self._dirac(k+70) - 876*self._dirac(k+71) - 980*self._dirac(k+72) - 1081*self._dirac(k+73) -
                        1179*self._dirac(k+74) - 1274*self._dirac(k+75) - 1366*self._dirac(k+76) - 1455*self._dirac(k+77) -
                        1541*self._dirac(k+78) - 1624*self._dirac(k+79) - 1704*self._dirac(k+80) - 1781*self._dirac(k+81) -
                        1855*self._dirac(k+82) - 1926*self._dirac(k+83) - 1994*self._dirac(k+84) - 2059*self._dirac(k+85) -
                        2121*self._dirac(k+86) - 2180*self._dirac(k+87) - 2236*self._dirac(k+88) - 2289*self._dirac(k+89) -
                        2339*self._dirac(k+90) - 2386*self._dirac(k+91) - 2430*self._dirac(k+92) - 2471*self._dirac(k+93) -
                        2509*self._dirac(k+94) - 2544*self._dirac(k+95) - 2576*self._dirac(k+96) - 2605*self._dirac(k+97) -
                        2631*self._dirac(k+98) - 2654*self._dirac(k+99) - 2674*self._dirac(k+100) - 2691*self._dirac(k+101) -
                        2705*self._dirac(k+102) - 2716*self._dirac(k+103) - 2724*self._dirac(k+104) - 2729*self._dirac(k+105) -
                        2731*self._dirac(k+106) - 2730*self._dirac(k+107) - 2726*self._dirac(k+108) - 2719*self._dirac(k+109) -
                        2709*self._dirac(k+110) - 2696*self._dirac(k+111) - 2680*self._dirac(k+112) - 2661*self._dirac(k+113) -
                        2639*self._dirac(k+114) - 2614*self._dirac(k+115) - 2586*self._dirac(k+116) - 2555*self._dirac(k+117) -
                        2521*self._dirac(k+118) - 2484*self._dirac(k+119) - 2444*self._dirac(k+120) - 2401*self._dirac(k+121) -
                        2355*self._dirac(k+122) - 2306*self._dirac(k+123) - 2254*self._dirac(k+124) - 2199*self._dirac(k+125) -
                        2141*self._dirac(k+126) - 2080*self._dirac(k+127) - 2016*self._dirac(k+128) - 1953*self._dirac(k+129) -
                        1891*self._dirac(k+130) - 1830*self._dirac(k+131) - 1770*self._dirac(k+132) - 1711*self._dirac(k+133) -
                        1653*self._dirac(k+134) - 1596*self._dirac(k+135) - 1540*self._dirac(k+136) - 1485*self._dirac(k+137) -
                        1431*self._dirac(k+138) - 1378*self._dirac(k+139) - 1326*self._dirac(k+140) - 1275*self._dirac(k+141) -
                        1225*self._dirac(k+142) - 1176*self._dirac(k+143) - 1128*self._dirac(k+144) - 1081*self._dirac(k+145) -
                        1035*self._dirac(k+146) - 990*self._dirac(k+147) - 946*self._dirac(k+148) - 903*self._dirac(k+149) -
                        861*self._dirac(k+150) - 820*self._dirac(k+151) - 780*self._dirac(k+152) - 741*self._dirac(k+153) -
                        703*self._dirac(k+154) - 666*self._dirac(k+155) - 630*self._dirac(k+156) - 595*self._dirac(k+157) -
                        561*self._dirac(k+158) - 528*self._dirac(k+159) - 496*self._dirac(k+160) - 465*self._dirac(k+161) -
                        435*self._dirac(k+162) - 406*self._dirac(k+163) - 378*self._dirac(k+164) - 351*self._dirac(k+165) -
                        325*self._dirac(k+166) - 300*self._dirac(k+167) - 276*self._dirac(k+168) - 253*self._dirac(k+169) -
                        231*self._dirac(k+170) - 210*self._dirac(k+171) - 190*self._dirac(k+172) - 171*self._dirac(k+173) -
                        153*self._dirac(k+174) - 136*self._dirac(k+175) - 120*self._dirac(k+176) - 105*self._dirac(k+177) -
                        91*self._dirac(k+178) - 78*self._dirac(k+179) - 66*self._dirac(k+180) - 55*self._dirac(k+181) -
                        45*self._dirac(k+182) - 36*self._dirac(k+183) - 28*self._dirac(k+184) - 21*self._dirac(k+185) -
                        15*self._dirac(k+186) - 10*self._dirac(k+187) - 6*self._dirac(k+188) - 3*self._dirac(k+189) -
                        self._dirac(k+190))
                
                coeff_val = (-1 / 8192 )* (s7p2+s7p1)
                        
            elif scale == 8:
                
                
                s8p1 = (self._dirac(k-127) + 3*self._dirac(k-126) + 6*self._dirac(k-125) + 10*self._dirac(k-124) + 15*self._dirac(k-123) +
                        21*self._dirac(k-122) + 28*self._dirac(k-121) + 36*self._dirac(k-120) + 45*self._dirac(k-119) + 55*self._dirac(k-118) +
                        66*self._dirac(k-117) + 78*self._dirac(k-116) + 91*self._dirac(k-115) + 105*self._dirac(k-114) + 120*self._dirac(k-113) +
                        136*self._dirac(k-112) + 153*self._dirac(k-111) + 171*self._dirac(k-110) + 190*self._dirac(k-109) + 210*self._dirac(k-108) +
                        231*self._dirac(k-107) + 253*self._dirac(k-106) + 276*self._dirac(k-105) + 300*self._dirac(k-104) + 325*self._dirac(k-103) +
                        351*self._dirac(k-102) + 378*self._dirac(k-101) + 406*self._dirac(k-100) + 435*self._dirac(k-99) + 465*self._dirac(k-98) +
                        496*self._dirac(k-97) + 528*self._dirac(k-96) + 561*self._dirac(k-95) + 595*self._dirac(k-94) + 630*self._dirac(k-93) +
                        666*self._dirac(k-92) + 703*self._dirac(k-91) + 741*self._dirac(k-90) + 780*self._dirac(k-89) + 820*self._dirac(k-88) +
                        861*self._dirac(k-87) + 903*self._dirac(k-86) + 946*self._dirac(k-85) + 990*self._dirac(k-84) + 1035*self._dirac(k-83) +
                        1081*self._dirac(k-82) + 1128*self._dirac(k-81) + 1176*self._dirac(k-80) + 1225*self._dirac(k-79) + 1275*self._dirac(k-78) +
                        1326*self._dirac(k-77) + 1378*self._dirac(k-76) + 1431*self._dirac(k-75) + 1485*self._dirac(k-74) + 1540*self._dirac(k-73) +
                        1596*self._dirac(k-72) + 1653*self._dirac(k-71) + 1711*self._dirac(k-70) + 1770*self._dirac(k-69) + 1830*self._dirac(k-68) +
                        1891*self._dirac(k-67) + 1953*self._dirac(k-66) + 2016*self._dirac(k-65) + 2080*self._dirac(k-64) + 2145*self._dirac(k-63) +
                        2211*self._dirac(k-62) + 2278*self._dirac(k-61) + 2346*self._dirac(k-60) + 2415*self._dirac(k-59) + 2485*self._dirac(k-58) +
                        2556*self._dirac(k-57) + 2628*self._dirac(k-56) + 2701*self._dirac(k-55) + 2775*self._dirac(k-54) + 2850*self._dirac(k-53) +
                        2926*self._dirac(k-52) + 3003*self._dirac(k-51) + 3081*self._dirac(k-50) + 3160*self._dirac(k-49) + 3240*self._dirac(k-48) +
                        3321*self._dirac(k-47) + 3403*self._dirac(k-46) + 3486*self._dirac(k-45) + 3570*self._dirac(k-44) + 3655*self._dirac(k-43) +
                        3741*self._dirac(k-42) + 3828*self._dirac(k-41) + 3916*self._dirac(k-40) + 4005*self._dirac(k-39) + 4095*self._dirac(k-38) +
                        4186*self._dirac(k-37) + 4278*self._dirac(k-36) + 4371*self._dirac(k-35) + 4465*self._dirac(k-34) + 4560*self._dirac(k-33) +
                        4656*self._dirac(k-32) + 4753*self._dirac(k-31) + 4851*self._dirac(k-30) + 4950*self._dirac(k-29) + 5050*self._dirac(k-28) +
                        5151*self._dirac(k-27) + 5253*self._dirac(k-26) + 5356*self._dirac(k-25) + 5460*self._dirac(k-24) + 5565*self._dirac(k-23) +
                        5671*self._dirac(k-22) + 5778*self._dirac(k-21) + 5886*self._dirac(k-20) + 5995*self._dirac(k-19) + 6105*self._dirac(k-18) +
                        6216*self._dirac(k-17) + 6328*self._dirac(k-16) + 6441*self._dirac(k-15) + 6555*self._dirac(k-14) + 6670*self._dirac(k-13) +
                        6786*self._dirac(k-12) + 6903*self._dirac(k-11) + 7021*self._dirac(k-10) + 7140*self._dirac(k-9) + 7260*self._dirac(k-8) +
                        7381*self._dirac(k-7) + 7503*self._dirac(k-6) + 7626*self._dirac(k-5) + 7750*self._dirac(k-4) + 7875*self._dirac(k-3))

                s8p2 = (8001*self._dirac(k-2) + 8128*self._dirac(k-1) + 8256*self._dirac(k) + 8381*self._dirac(k+1) + 8503*self._dirac(k+2) +
                        8622*self._dirac(k+3) + 8738*self._dirac(k+4) + 8851*self._dirac(k+5) + 8961*self._dirac(k+6) + 9068*self._dirac(k+7) +
                        9172*self._dirac(k+8) + 9273*self._dirac(k+9) + 9371*self._dirac(k+10) + 9466*self._dirac(k+11) + 9558*self._dirac(k+12) +
                        9647*self._dirac(k+13) + 9733*self._dirac(k+14) + 9816*self._dirac(k+15) + 9896*self._dirac(k+16) + 9973*self._dirac(k+17) +
                        10047*self._dirac(k+18) + 10118*self._dirac(k+19) + 10186*self._dirac(k+20) + 10251*self._dirac(k+21) + 10313*self._dirac(k+22) +
                        10372*self._dirac(k+23) + 10428*self._dirac(k+24) + 10481*self._dirac(k+25) + 10531*self._dirac(k+26) + 10578*self._dirac(k+27) +
                        10622*self._dirac(k+28) + 10663*self._dirac(k+29) + 10701*self._dirac(k+30) + 10736*self._dirac(k+31) + 10768*self._dirac(k+32) +
                        10797*self._dirac(k+33) + 10823*self._dirac(k+34) + 10846*self._dirac(k+35) + 10866*self._dirac(k+36) + 10883*self._dirac(k+37) +
                        10897*self._dirac(k+38) + 10908*self._dirac(k+39) + 10916*self._dirac(k+40) + 10921*self._dirac(k+41) + 10923*self._dirac(k+42) +
                        10922*self._dirac(k+43) + 10918*self._dirac(k+44) + 10911*self._dirac(k+45) + 10901*self._dirac(k+46) + 10888*self._dirac(k+47) +
                        10872*self._dirac(k+48) + 10853*self._dirac(k+49) + 10831*self._dirac(k+50) + 10806*self._dirac(k+51) + 10778*self._dirac(k+52) +
                        10747*self._dirac(k+53) + 10713*self._dirac(k+54) + 10676*self._dirac(k+55) + 10636*self._dirac(k+56) + 10593*self._dirac(k+57) +
                        10547*self._dirac(k+58) + 10498*self._dirac(k+59) + 10446*self._dirac(k+60) + 10391*self._dirac(k+61) + 10333*self._dirac(k+62) +
                        10272*self._dirac(k+63) + 10208*self._dirac(k+64) + 10141*self._dirac(k+65) + 10071*self._dirac(k+66) + 9998*self._dirac(k+67) +
                        9922*self._dirac(k+68) + 9843*self._dirac(k+69) + 9761*self._dirac(k+70) + 9676*self._dirac(k+71) + 9588*self._dirac(k+72) +
                        9497*self._dirac(k+73) + 9403*self._dirac(k+74) + 9306*self._dirac(k+75) + 9206*self._dirac(k+76) + 9103*self._dirac(k+77) +
                        8997*self._dirac(k+78) + 8888*self._dirac(k+79) + 8776*self._dirac(k+80) + 8661*self._dirac(k+81) + 8543*self._dirac(k+82) +
                        8422*self._dirac(k+83) + 8298*self._dirac(k+84) + 8171*self._dirac(k+85) + 8041*self._dirac(k+86) + 7908*self._dirac(k+87) +
                        7772*self._dirac(k+88) + 7633*self._dirac(k+89) + 7491*self._dirac(k+90) + 7346*self._dirac(k+91) + 7198*self._dirac(k+92) +
                        7047*self._dirac(k+93) + 6893*self._dirac(k+94) + 6736*self._dirac(k+95) + 6576*self._dirac(k+96) + 6413*self._dirac(k+97) +
                        6247*self._dirac(k+98) + 6078*self._dirac(k+99) + 5906*self._dirac(k+100) + 5731*self._dirac(k+101) + 5553*self._dirac(k+102) +
                        5372*self._dirac(k+103) + 5188*self._dirac(k+104) + 5001*self._dirac(k+105) + 4811*self._dirac(k+106) + 4618*self._dirac(k+107) +
                        4422*self._dirac(k+108) + 4223*self._dirac(k+109) + 4021*self._dirac(k+110) + 3816*self._dirac(k+111) + 3608*self._dirac(k+112) +
                        3397*self._dirac(k+113) + 3183*self._dirac(k+114) + 2966*self._dirac(k+115) + 2746*self._dirac(k+116) + 2523*self._dirac(k+117) +
                        2297*self._dirac(k+118) + 2068*self._dirac(k+119) + 1836*self._dirac(k+120) + 1601*self._dirac(k+121) + 1363*self._dirac(k+122) +
                        1122*self._dirac(k+123) + 878*self._dirac(k+124) + 631*self._dirac(k+125) + 381*self._dirac(k+126) + 128*self._dirac(k+127)) 
                
                s8p3 = (- 128*self._dirac(k+128) - 381*self._dirac(k+129) - 631*self._dirac(k+130) - 878*self._dirac(k+131) - 1122*self._dirac(k+132) -
                        1363*self._dirac(k+133) - 1601*self._dirac(k+134) - 1836*self._dirac(k+135) - 2068*self._dirac(k+136) - 2297*self._dirac(k+137) -
                        2523*self._dirac(k+138) - 2746*self._dirac(k+139) - 2966*self._dirac(k+140) - 3183*self._dirac(k+141) - 3397*self._dirac(k+142) -
                        3608*self._dirac(k+143) - 3816*self._dirac(k+144) - 4021*self._dirac(k+145) - 4223*self._dirac(k+146) - 4422*self._dirac(k+147) -
                        4618*self._dirac(k+148) - 4811*self._dirac(k+149) - 5001*self._dirac(k+150) - 5188*self._dirac(k+151) - 5372*self._dirac(k+152) -
                        5553*self._dirac(k+153) - 5731*self._dirac(k+154) - 5906*self._dirac(k+155) - 6078*self._dirac(k+156) - 6247*self._dirac(k+157) -
                        6413*self._dirac(k+158) - 6576*self._dirac(k+159) - 6736*self._dirac(k+160) - 6893*self._dirac(k+161) - 7047*self._dirac(k+162) -
                        7198*self._dirac(k+163) - 7346*self._dirac(k+164) - 7491*self._dirac(k+165) - 7633*self._dirac(k+166) - 7772*self._dirac(k+167) -
                        7908*self._dirac(k+168) - 8041*self._dirac(k+169) - 8171*self._dirac(k+170) - 8298*self._dirac(k+171) - 8422*self._dirac(k+172) -
                        8543*self._dirac(k+173) - 8661*self._dirac(k+174) - 8776*self._dirac(k+175) - 8888*self._dirac(k+176) - 8997*self._dirac(k+177) -
                        9103*self._dirac(k+178) - 9206*self._dirac(k+179) -
                        9306*self._dirac(k+180) - 9403*self._dirac(k+181) - 9497*self._dirac(k+182) - 9588*self._dirac(k+183) - 9676*self._dirac(k+184) -
                        9761*self._dirac(k+185) - 9843*self._dirac(k+186) - 9922*self._dirac(k+187) - 9998*self._dirac(k+188) - 10071*self._dirac(k+189) -
                        10141*self._dirac(k+190) - 10208*self._dirac(k+191) - 10272*self._dirac(k+192) - 10333*self._dirac(k+193) - 10391*self._dirac(k+194) -
                        10446*self._dirac(k+195) - 10498*self._dirac(k+196) - 10547*self._dirac(k+197) - 10593*self._dirac(k+198) - 10636*self._dirac(k+199) -
                        10676*self._dirac(k+200) - 10713*self._dirac(k+201) - 10747*self._dirac(k+202) - 10778*self._dirac(k+203) - 10806*self._dirac(k+204) -
                        10831*self._dirac(k+205) - 10853*self._dirac(k+206) - 10872*self._dirac(k+207) - 10888*self._dirac(k+208) - 10901*self._dirac(k+209) -
                        10911*self._dirac(k+210) - 10918*self._dirac(k+211) - 10922*self._dirac(k+212) - 10923*self._dirac(k+213) - 10921*self._dirac(k+214) -
                        10916*self._dirac(k+215) - 10908*self._dirac(k+216) - 10897*self._dirac(k+217) - 10883*self._dirac(k+218) - 10866*self._dirac(k+219) -
                        10846*self._dirac(k+220) - 10823*self._dirac(k+221) - 10797*self._dirac(k+222) - 10768*self._dirac(k+223) - 10736*self._dirac(k+224) -
                        10701*self._dirac(k+225) - 10663*self._dirac(k+226) - 10622*self._dirac(k+227) - 10578*self._dirac(k+228) - 10531*self._dirac(k+229) -
                        10481*self._dirac(k+230) - 10428*self._dirac(k+231) - 10372*self._dirac(k+232) - 10313*self._dirac(k+233) - 10251*self._dirac(k+234) -
                        10186*self._dirac(k+235) - 10118*self._dirac(k+236) - 10047*self._dirac(k+237) - 9973*self._dirac(k+238) - 9896*self._dirac(k+239) )

                s8p4 = (- 9816*self._dirac(k+240) - 9733*self._dirac(k+241) - 9647*self._dirac(k+242) - 9558*self._dirac(k+243) - 9466*self._dirac(k+244) -
                        9371*self._dirac(k+245) - 9273*self._dirac(k+246) - 9172*self._dirac(k+247) - 9068*self._dirac(k+248) - 8961*self._dirac(k+249) -
                        8851*self._dirac(k+250) - 8738*self._dirac(k+251) - 8622*self._dirac(k+252) - 8503*self._dirac(k+253) - 8381*self._dirac(k+254) -
                        8256*self._dirac(k+255) - 8128*self._dirac(k+256) - 8001*self._dirac(k+257) - 7875*self._dirac(k+258) - 7750*self._dirac(k+259) -
                        7626*self._dirac(k+260) - 7503*self._dirac(k+261) - 7381*self._dirac(k+262) - 7260*self._dirac(k+263) - 7140*self._dirac(k+264) -
                        7021*self._dirac(k+265) - 6903*self._dirac(k+266) - 6786*self._dirac(k+267) - 6670*self._dirac(k+268) - 6555*self._dirac(k+269) -
                        6441*self._dirac(k+270) - 6328*self._dirac(k+271) - 6216*self._dirac(k+272) - 6105*self._dirac(k+273) - 5995*self._dirac(k+274) -
                        5886*self._dirac(k+275) - 5778*self._dirac(k+276) - 5671*self._dirac(k+277) - 5565*self._dirac(k+278) - 5460*self._dirac(k+279) -
                        5356*self._dirac(k+280) - 5253*self._dirac(k+281) - 5151*self._dirac(k+282) - 5050*self._dirac(k+283) - 4950*self._dirac(k+284) -
                        4851*self._dirac(k+285) - 4753*self._dirac(k+286) - 4656*self._dirac(k+287) - 4560*self._dirac(k+288) - 4465*self._dirac(k+289) -
                        4371*self._dirac(k+290) - 4278*self._dirac(k+291) - 4186*self._dirac(k+292) - 4095*self._dirac(k+293) - 4005*self._dirac(k+294) -
                        3916*self._dirac(k+295) - 3828*self._dirac(k+296) - 3741*self._dirac(k+297) - 3655*self._dirac(k+298) - 3570*self._dirac(k+299) -
                        3486*self._dirac(k+300) - 3403*self._dirac(k+301) - 3321*self._dirac(k+302) - 3240*self._dirac(k+303) - 3160*self._dirac(k+304) -
                        3081*self._dirac(k+305) - 3003*self._dirac(k+306) - 2926*self._dirac(k+307) - 2850*self._dirac(k+308) - 2775*self._dirac(k+309) -
                        2701*self._dirac(k+310) - 2628*self._dirac(k+311) - 2556*self._dirac(k+312) - 2485*self._dirac(k+313) - 2415*self._dirac(k+314) -
                        2346*self._dirac(k+315) - 2278*self._dirac(k+316) - 2211*self._dirac(k+317) - 2145*self._dirac(k+318) - 2080*self._dirac(k+319) -
                        2016*self._dirac(k+320) - 1953*self._dirac(k+321) - 1891*self._dirac(k+322) - 1830*self._dirac(k+323) - 1770*self._dirac(k+324) -
                        1711*self._dirac(k+325) - 1653*self._dirac(k+326) - 1596*self._dirac(k+327) - 1540*self._dirac(k+328) - 1485*self._dirac(k+329) -
                        1431*self._dirac(k+330) - 1378*self._dirac(k+331) - 1326*self._dirac(k+332) - 1275*self._dirac(k+333) - 1225*self._dirac(k+334) -
                        1176*self._dirac(k+335) - 1128*self._dirac(k+336) - 1081*self._dirac(k+337) - 1035*self._dirac(k+338) - 990*self._dirac(k+339) -
                        946*self._dirac(k+340) - 903*self._dirac(k+341) - 861*self._dirac(k+342) - 820*self._dirac(k+343) - 780*self._dirac(k+344) -
                        741*self._dirac(k+345) - 703*self._dirac(k+346) - 666*self._dirac(k+347) - 630*self._dirac(k+348) - 595*self._dirac(k+349) -
                        561*self._dirac(k+350) - 528*self._dirac(k+351) - 496*self._dirac(k+352) - 465*self._dirac(k+353) - 435*self._dirac(k+354) -
                        406*self._dirac(k+355) - 378*self._dirac(k+356) - 351*self._dirac(k+357) - 325*self._dirac(k+358) - 300*self._dirac(k+359) -
                        276*self._dirac(k+360) - 253*self._dirac(k+361) - 231*self._dirac(k+362) - 210*self._dirac(k+363) - 190*self._dirac(k+364) -
                        171*self._dirac(k+365) - 153*self._dirac(k+366) - 136*self._dirac(k+367) - 120*self._dirac(k+368) - 105*self._dirac(k+369) -
                        91*self._dirac(k+370) - 78*self._dirac(k+371) - 66*self._dirac(k+372) - 55*self._dirac(k+373) - 45*self._dirac(k+374) -
                        36*self._dirac(k+375) - 28*self._dirac(k+376) - 21*self._dirac(k+377) - 15*self._dirac(k+378) - 10*self._dirac(k+379) -
                        6*self._dirac(k+380) - 3*self._dirac(k+381) - self._dirac(k+382))
                
                coeff_val = (s8p1 +s8p2+ s8p3 +s8p4 ) * (-1/32768)
            
            else:
                raise ValueError("Scale must be between 1 and 8.")
            

            if coeff_val != 0:
                qj[k + abs(a)] = coeff_val
        
        return qj