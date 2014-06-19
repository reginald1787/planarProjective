import numpy as np
from scipy.linalg import sqrtm 
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import cv2
import cv

  


def cubic(P,nP):
    cov = np.zeros(4)
    for i in range(nP): 
        cov[0] += P[0,i]**2 
        cov[3] += P[1,i]**2
        cov[1] += P[0,i]*P[1,i] 
     
    cov[2] = cov[1] 

    eign,eigv = Eigen2by2Sym(cov) 
    eign[0] = 1.0/np.sqrt(eign[0]) 
    eign[1] = 1.0/np.sqrt(eign[1])  

    S = np.zeros(4)

    S[0] = eigv[0]*eigv[0]*eign[0] + eigv[1]*eigv[1]*eign[1] 
    S[1] = eigv[0]*eigv[2]*eign[0] + eigv[1]*eigv[3]*eign[1] 
    S[2] = S[1] 
    S[3] = eigv[2]*eigv[2]*eign[0] + eigv[3]*eigv[3]*eign[1] 

    # cubic moment of P2
    cubicM = np.zeros(4)
    for i in range(nP): 
        cubicM[0] += pow(P[0,i],3) 
        cubicM[1] += pow(P[0,i],2)*P[1,i] 
        cubicM[2] += P[0,i]*pow(P[1,i],2) 
        cubicM[3] += pow(P[1,i],3) 
     
    #for(int i=0 i<4 i++)
    for i in range(4):
        cubicM[i] = cubicM[i]/float(nP)

    return S,cubicM 

# C = A*B
def matrixMult2x2(A, B):
    C = np.zeros(4)
    C[0] = A[0]*B[0] + A[1]*B[2] 
    C[1] = A[0]*B[1] + A[1]*B[3] 
    C[2] = A[2]*B[0] + A[3]*B[2] 
    C[3] = A[2]*B[1] + A[3]*B[3] 
    return C

def Inverse2x2(mat):
 
    det = mat[0]*mat[3] - mat[1]*mat[2] 
    inv = np.zeros(4)

    if(det == 0):# inverse of mat doesn't exist. 
     
        inv[0] = 0         
        inv[1] = 0 
        inv[2] = 0         
        inv[3] = 0 

    else: 
	    inv[0] = mat[3]/det 
	    inv[1] = -1*mat[1]/det 
	    inv[2] = -1*mat[2]/det 
	    inv[3] = mat[0]/det 

    return inv

 
def Eigen2by2Sym(Matrix):
 
    eigenvalue = np.zeros(2)
    eigenvectors = np.zeros(4)

    a=Matrix[0] 
    c=Matrix[1] 
    b=Matrix[3] 
    B=-a-b 
    C=a*b-c*c 
    e1=(-B+ np.sqrt(B*B-4.0 *C))/2.0  
    e2=(-B- np.sqrt(B*B-4.0 *C))/2.0  

    if(e2 < e1):
     
        B=e1 
        e1=e2 
        e2=B 
     
    eigenvalue[0]=e1 
    eigenvalue[1]=e2 
    C= np.sqrt((c*c+(e1-a)*(e1-a))) 
    if(C < 0.00001):
      #/ so c is very close to zero ....  
        eigenvectors[0]=1.0 
        eigenvectors[1]=0.0 
        eigenvectors[2]=0.0 
        eigenvectors[3]=1.0 
     
    else:     
	    eigenvectors[0]=c/C 
	    eigenvectors[1]=(e1-a)/C 
	    eigenvectors[2]=(eigenvectors[1]) 
	    eigenvectors[3]=-eigenvectors[0] 

    return eigenvalue,eigenvectors
 

def Normalize(p,nP):
 

    mean = np.mean(p,axis=1)
     
    
    dP = p - mean.reshape(2,1)
    maxX = dP.max(axis=1)[0]
    maxY = dP.max(axis=1)[1]
    minX = dP.min(axis=1)[0]
    minY = dP.min(axis=1)[1]


    # doing the normalization here
    nX = max(abs(maxX),abs(minX)) 
    nY = max(abs(maxY),abs(minY)) 

    dP  = dP/np.array([nX,nY]).reshape(2,1)

    return dP, np.array([nX,nY]), mean 
#---------------------------------------------------------------------------------------------yy
def CheckQuality(mat2x2, p1, nP1, p2, nP2, intv):
 


    # transform points set 1
    tp1 = np.zeros((2,nP1))
    for i in range(nP1):
     
        tp1[0,i]  = mat2x2[0]*p1[0,i] + mat2x2[1]*p1[1,i] 
        tp1[1,i]= mat2x2[2]*p1[0,i] + mat2x2[3]*p1[1,i] 
     


    value = 0.0 
    mD = 9999999.0 
    maxD = -100.0 
    dist = 0.0

    for i in range(0,nP1,intv):
     
        mD = 99999 
        for j in range(0,nP2,intv):
         
            dist = (tp1[0,i] - p2[0,j])*(tp1[0,i] - p2[0,j] ) + (tp1[1,i]-p2[1,j])*(tp1[1,i]-p2[1,j]) 
            if(dist < mD):
                mD = dist 
         
        if(mD > maxD):
            maxD = mD  
     

    #mexPrintf("HELLO!!\n\n\n\n") 
    maxD2 = -100.0 

    for i in range(0,nP2,intv):

     
        mD = 99999 
        for j in range(0,nP1,intv):
         
            dist = (tp1[0,j] - p2[0,i])*  (tp1[0,j] - p2[0,i] ) + (tp1[1,j]-p2[1,i])*(tp1[1,j]-p2[1,i]) 
            if(dist < mD):
                mD = dist 
         
        if(mD > maxD2):
            maxD2 = mD  
     
    
    #w
    #mexPrintf("\n\n\n HELLO 2 2 2 2 \n\n\n") 


    return np.sqrt(maxD)+np.sqrt(maxD2) 

 
#---------------------------------------------------------------------------------------------yy
def GetAffine2D(p1, p2, nP1, nP2, S2, cubicM2, dScale,angles, nAngles):
 
    mean2= [0,0]  
    
    tP2 = p2 


    nMins = 0 

    
    mean1 = np.mean(p1,axis=1)
     
    tP1 = p1 - mean1.reshape(2,1)
   
    S1,cubicM1 = cubic(tP1,nP1)


    # W2 = inv(S2)
    W2 = Inverse2x2(S2) 
    #for(int i=0 i<4 i++) 
    W1 = np.zeros(4)
    for i in range(4): 
        W1[i] = S1[i] 

    #mexPrintf("\nW2:(%f,%f,%f,%f)",W2[0],W2[1],W2[2],W2[3]) 
    #mexPrintf("\nW1:(%f,%f,%f,%f)\n",W1[0],W1[1],W1[2],W1[3]) 

    # Calculate A, B, C, D
    #for(int i=0 i<nAngles*2 i++)
    A = np.zeros(2*nAngles)
    B = np.zeros(2*nAngles)
    C = np.zeros(2*nAngles)
    D = np.zeros(2*nAngles)

    for i in range(2*nAngles):

        A[i] = angles[4*i]*W2[0]*W1[0] + angles[4*i+1]*W2[0]*W1[1] + angles[4*i+2]*W2[1]*W1[0] + angles[4*i+3]*W2[1]*W1[1] 
        B[i] = angles[4*i]*W2[0]*W1[2] + angles[4*i+1]*W2[0]*W1[3] + angles[4*i+2]*W2[1]*W1[2] + angles[4*i+3]*W2[1]*W1[3] 
        C[i] = angles[4*i]*W2[2]*W1[0] + angles[4*i+1]*W2[2]*W1[1] + angles[4*i+2]*W2[3]*W1[0] + angles[4*i+3]*W2[3]*W1[1] 
        D[i] = angles[4*i]*W2[2]*W1[2] + angles[4*i+1]*W2[2]*W1[3] + angles[4*i+2]*W2[3]*W1[2] + angles[4*i+3]*W2[3]*W1[3] 
     

    
    
    # calculting MM.
    MM = np.zeros(8*nAngles)
    GX = np.zeros(2*nAngles)
    for i in range(2*nAngles):
     
        MM[4*i]     = pow(A[i],3)*cubicM1[0]    + 3*pow(A[i],2)*B[i] * cubicM1[1] + 3*A[i]*B[i]*B[i]*cubicM1[2] + pow(B[i],3)*cubicM1[3] 
        MM[4*i+1]   = A[i]*A[i]*C[i]*cubicM1[0] + (2*A[i]*B[i]*C[i]+A[i]*A[i]*D[i])*cubicM1[1] + (2*A[i]*B[i]*D[i]+B[i]*B[i]*C[i])*cubicM1[2] + B[i]*B[i]*D[i]*cubicM1[3] 
        MM[4*i+2]   = C[i]*C[i]*A[i]*cubicM1[0] + (2*C[i]*D[i]*A[i]+C[i]*C[i]*B[i])*cubicM1[1] + (2*C[i]*D[i]*B[i]+D[i]*D[i]*A[i])*cubicM1[2] + D[i]*D[i]*B[i]*cubicM1[3] 
        MM[4*i+3]   = pow(C[i],3)*cubicM1[0]    + 3*C[i]*C[i]*D[i]*cubicM1[1]     + 3*C[i]*D[i]*D[i]*cubicM1[2] + pow(D[i],3)*cubicM1[3] 

        # this is the part that calculate KX. 
        MM[4*i]     -= cubicM2[0] 
        MM[4*i+1]   -= cubicM2[1] 
        MM[4*i+2]   -= cubicM2[2] 
        MM[4*i+3]   -= cubicM2[3] 
        
        #if(i>nAngles*2-4):
                
            #mexPrintf("KX[%d]=%.8f,KX[%d]=%.8f,KX[%d]=%.8f,KX[%d]=%.8f,\n",4*i,MM[4*i],4*i+1,MM[4*i+1],4*i+2,MM[4*i+2],4*i+3,MM[4*i+3]) 
         
        # calculate GX
        
        GX[i] = pow( (MM[4*i]/(abs(cubicM2[0])+0.0001)),2) + pow((MM[4*i+1]/(abs(cubicM2[1])+0.0001)),2) + pow((MM[4*i+2]/(abs(cubicM2[2])+0.0001)),2) + pow((MM[4*i+3]/(abs(cubicM2[3])+0.0001)),2)  
     

   
    minV = 9999999 
    minI = -1 
    tempidx1 = 2 
    head = [[0,9999999] for i in range(4)]

    #for(int i=1 i<nAngles*2-1 i++)
    for i in range(1,2*nAngles-1): 

        if(GX[i]<=GX[i-1] and GX[i]<=GX[i+1] and GX[i] < head[2][1] ):
           
            tempidx1 = 2 
            #for(int j=1 j>=0 j--)
            for j in range(1,-1,-1):
                if(GX[i] < head[j][1]):
                   tempidx1 = j  
               
            #for(int j=2 j>= j++)
            #
            #
            if(tempidx1 ==2):
             
                head[tempidx1][1] = GX[i] 
                head[tempidx1][0] = i 
             
            elif(tempidx1 == 1):
             
                head[2][1] = head[1][1] 
                head[2][0] = head[1][0] 
                #
                head[1][1] = GX[i] 
                head[1][0] = i 
             
            elif(tempidx1 == 0):
             
                
                head[2][1] = head[1][1] 
                head[2][0] = head[1][0] 
                #
                head[1][1] = head[0][1] 
                head[1][0] = head[0][0] 

                head[0][1] = GX[i] 
                head[0][0] = i 
             
         
     

    #minValues[0] = minV 
    #minIndex[0] = minI 
    nMins = 3 

    # theta 
    # RR[4], iS2[4], RRS1[4] 
    # AA[4] 
    #minValue = 9999999#,value 


    
    # identity matrix
    AA = [1,0,0,1]  
    

    #for(int i=0 i<4 i++)
    resultA2x2 = [1,0,0,1]

    # for i in range(4):
    #     resultA2x2[i] = AA[i] 
    
    minValue = CheckQuality(AA, tP1, nP1, tP2, nP2,dScale) 
    
    

    RR = np.zeros(4)
    #for(int i=0 i<nMins i++)
    for i in range(nMins): 

        # # build the rotational matrix
        # # get theta
        # theta = angles[head[i].m_Index] 

        # # build the rotational matrix
        # RR[0] = cos(theta)  RR[1] = -sin(theta)  RR[2] = sin(theta)  RR[3] = cos(theta) 
        # # if it's a reflection
        index11 = head[i][0] 
        #for(int k=0 k<4 k++)
        for k in range(4):
            RR[k] = angles[4*index11+k] 


        if(head[i][0] > nAngles):
         
            RR[2] = -1*RR[2] 
            RR[3] = -1*RR[3] 

        # AA = inv(S2)*RR*S1 
        RRS1 = matrixMult2x2(RR,S1) 
        iS2 = Inverse2x2(S2) 
        #mexPrintf("Theta = %.4f\n",theta) 
        #mexPrintf("RR =(%.4f,%.4f,%.4f,%.4f )\n",RR[0],RR[1],RR[2],RR[3]) 
        #mexPrintf("inverseS2=(%.3f,%.3f,%.3f,%.3f )\n",iS2[0],iS2[1],iS2[2],iS2[3]) 
        #mexPrintf("RRS1=(%.3f,%.3f,%.3f,%.3f )\n",RRS1[0],RRS1[1],RRS1[2],RRS1[3]) 

        # AA here is Affine2x2(if criteria satisfies.....)
        AA = matrixMult2x2(iS2,RRS1) 
        
        value = CheckQuality(AA,tP1, nP1, tP2, nP2, dScale) 
        #
        #mexPrintf("[%.4f\t%.4f]\n",AA[0], AA[1]) 
        #mexPrintf("[%.4f\t%.4f]\n",AA[2], AA[3]) 
        #mexPrintf("Quality %d is %.4f\n",i,value) 
        if(value < minValue):
         
            minValue = value 
            #for(int j=0 j<4 j++)
            for j in range(4):
                resultA2x2[j] = AA[j] 
           # mexPrintf("Min values found. The index is %d.\n", head[i].m_Index) 
         
     



    tMean = [0,0] 
    resultTT = [0,0]
    tMean[0] = resultA2x2[0]*mean1[0] + resultA2x2[1]*mean1[1] 
    tMean[1] = resultA2x2[2]*mean1[0] + resultA2x2[3]*mean1[1] 

    resultTT[0] = mean2[0] - tMean[0] 
    resultTT[1] = mean2[1] - tMean[1] 
    return resultA2x2,resultTT,minValue
    
 

def mexFunction(nP1, nP2, uP1, uP2, range1 = [-2,1,2], range2 = [-2,1,2], dint = 3):
 
    
    
    nAngles = int(360.0/0.1) 
    d_inc = 0.1/180.0*3.14159265 
    theta = 0 

    RMatrix = np.zeros(nAngles*4*2) 
  

    for i in range(nAngles): 
        RMatrix[4*i  ] = np.cos(theta)          
        RMatrix[4*i+1] = -np.sin(theta)          
        RMatrix[4*i+2] = -RMatrix[4*i+1]          
        RMatrix[4*i+3] = RMatrix[4*i]
        theta+=d_inc 
        #if(i<20)
          #  mexPrintf("cos(%.5f)=%.4f  ", theta,RMatrix[4*i]) 
     
    #mexPrintf("LOOK_++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n") 
    # reflection part
    #for(int i=nAngles i<2*nAngles i++)
    for i in range(nAngles,2*nAngles):  
        RMatrix[4*i  ] = RMatrix[4*(i-nAngles)] 
        RMatrix[4*i+1] = RMatrix[4*(i-nAngles)+1] 
        RMatrix[4*i+2] = -1*RMatrix[4*(i-nAngles)+2] 
        RMatrix[4*i+3] = -1*RMatrix[4*(i-nAngles)+3] 
     
     

  
    P1, HH1, TT1 = Normalize(uP1, nP1) 
    P2, HH2, TT2 = Normalize(uP2, nP2) 

    # Calculate the covariance of normalized points set 2
    #for(int i=0 i<nP2 i++)
    S2,cubicM2 = cubic(P2,nP2)



    # counter 
    count = 0 
    # the denominator to calculte the H(p)
    #float *aa = new float[nP1] 
    minD = 999999 
    #float rAA[4], rTT[2], rValue 
    minValue = 999999 
    minAA = np.zeros(4)
    minTT = np.zeros(2)
    ghw = np.zeros(2)

   
    for r1 in np.arange(range1[0],range1[2],range1[1]): 
        #for(float r2 = range2[0]  r2 <= range2[2]  r2 += range2[1])
        for r2 in np.arange(range2[0],range2[2],range2[1]):
            count +=1 
            # calculate the denominator
            minD = 999999 
            #memset(aa,0,sizeof*nP1)
            aa = np.zeros(nP1) 
            for i in range(nP1):
             
                aa[i] = abs(r1*P1[0,i] + r2*P1[1,i] + 1) 
                if(aa[i] < minD):
                    minD = aa[i]  
          
            if(minD > 0.0001):
             
                newP1 = P1/aa
                 
                # put GetAffine2D here

                
                rAA, rTT, rValue =  GetAffine2D(newP1, P2, nP1, nP2, S2, cubicM2, dint,RMatrix,nAngles) 
                
               

                if(rValue < minValue): 
                 
                    minValue = rValue 
                    #for(int i=0 i<4 i++)
                    for i in range(4):
                        minAA[i] = rAA[i] 
                        #mexPrintf("minAA[%d]=%.5f",i,minAA[i]) 
                    print r1,r2,minValue 
                    minTT[0] = rTT[0]  
                    minTT[1] = rTT[1] 
                    ghw[0] = r1  
                    ghw[1] = r2 
 
    # returning affine matrix
    #
    a= np.zeros(9)
    b=np.zeros(9) 
    finalMatrix= [0.0, 0.0, 0.0,   0.0, 0.0, 0.0,  0.0,0.0,0.0]  
    a[0] = minAA[0] 
    a[1] = minAA[1] 
    a[2] = minTT[0] 
    a[3] = minAA[2] 
    a[4] = minAA[3] 
    a[5] = minTT[1] 
    a[6] = 0.0      
    a[7] = 0.0      
    a[8] = 1.0 

    b[0] = 1.0      
    b[1] = 0.0      
    b[2] = 0.0 
    b[3] = 0.0      
    b[4] = 1.0     
    b[5] = 0.0 
    b[6] = ghw[0]   
    b[7] = ghw[1]  
    b[8] = 1.0    

    ptr = np.zeros(9)
    # mexPrintf("\n") 
    #for(int i=0 i<3 i++)
    for i in range(3):
        #for(int j=0 j<3 j++)
        for j in range(3): 
            #for(int k=0 k<3 k++)
            for k in range(3):
                finalMatrix[i*3+j] += a[i*3+k]*b[(k)*3+j] 
            #mexPrintf("%.4f\t",finalMatrix[i*3+j]) 
         
        #mexPrintf("\n") 
     
    #mexPrintf("\n") 

    # normalize ptr
    #for(int i=0 i<9 i++)
    for i in range(9): 
        ptr[i] = finalMatrix[i]/finalMatrix[8] 
        #mexPrintf("ptr[%d]=%.5f \n",i,finalMatrix[i]) 
     

    
    ptr1 = np.zeros((3,nP1)) 
    for i in range(nP1):
     
        #for(int j=0 j<3 j++)
        for j in range(3): 
            #for(int k=0 k<2 k++)
            for k in range(2): 
                ptr1[j,i] += finalMatrix[3*j+k]*P1[k,i]  
             
            ptr1[j,i] +=finalMatrix[3*j+2]*1 
         
        ptr1[0,i] = ptr1[0,i]/ptr1[2,i] 
        ptr1[1,i] = ptr1[1,i]/ptr1[2,i] 
        ptr1[2,i] = 1.0 
        
    

    return  ptr,HH1,TT1,HH2,TT2,ptr1
 



def plotdata(nP,nQ,HP):
    plt.subplot(2,2,1)
    plt.plot(nP[0,:],nP[1,:],'.',color='c')
    plt.title('P')
    
    plt.subplot(2,2,2)
    plt.plot(nQ[0,:],nQ[1,:],'.',color='g')
    plt.title('Q')

    # HP = project(H,P)
    #HP = minA.dot(nP)
    plt.subplot(2,2,3)
    plt.plot(HP[0,:],HP[1,:],'.')
    plt.title('H(P)')

    plt.show()

def readdata():
    data = loadmat('data/data_12.mat')
    print type(data),data.keys()
    P1 = data['P1']
    P2 = data['P2']
    img1 = data['img1c']
    img2 = data['img2c']
    print type(img1),img1.shape,type(img2),img2.shape
    print P1.shape,P2.shape
    # plt.subplot(1,2,1)
    # plt.imshow(img1)
    # plt.subplot(1,2,2)
    # plt.imshow(img2)
    # plt.show()
    return P1,P2

def main():
    
    P1,P2 = readdata()
    #plotdata(P1,P2,P1)
    n = P1.shape[1]
    
    H, S1 ,T1, S2, T2 ,tP1  = mexFunction(n, n, P1, P2) 

    nP1 = tP1[:2,:]*S1.reshape(2,1) + T1.reshape(2,1)
    #nP1 = tP1[:2,:]

    print nP1.shape
    plotdata(P1,P2,nP1)

if __name__ == '__main__':
    main()
