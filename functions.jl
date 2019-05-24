
using LinearAlgebra, SparseArrays, DelimitedFiles
#Specifying the number of particles and the interaction matrices 
N=3
jmat = zeros(Int8, N, N, 3);

#function to remove multiple elements from a list
function PopMul(list,rem) #rem = which elements to remove
    rem=sort(rem, rev=true) #ordering is important when removing elements from list, could be done elsewere
    for r in rem
       splice!(list,r) 
    end
    list
end

#function to convert a binary list [0,1,1,0,1] to a decimal number
function BinDec(bin)
    dec=0
    for i in 1:length(bin)
        dec+=bin[i]*2^(i-1)
    end
    Int64(dec)
end

function MakeHam(jmat)
    ham = spzeros(Float64,2^N, 2^N) #create empty sparse
    for n in 0:2^N-1 #going trhough all possible states n
        nbin=digits(n, base = 2, pad = N) #finding the binary corresponding to n (the state)
        for i in 1:N #i,j going through the elements of the J-matrices
            for j in i+1:N
                zi=nbin[i] #the ith number of the binary number (0 or 1) 
                zj=nbin[j]
                l = n +(1-2zi)*2^(i-1)+(1-2zj)*2^(j-1)
                ham[l+1,n+1] += (1/4)*jmat[i,j,1] #x-link
                ham[l+1,n+1] += -(1/4)*jmat[i,j,2]*(2zi-1)*(2zj-1) #y-link
                ham[n+1,n+1] += jmat[i,j,3]*(2zi-1)*(2zj-1) #z-link
            end
        end
    end
    return ham
end

function RedDens(state,subsys) #computes the reduced density matrix
    
    N=Int64(log(2,length(state))) #number of particles
    Na=length(subsys) #number of particles in subsystem A
    prod = zeros(Float64,2^Na, 2^(N-Na)) #matrix with coef. of the product state
    for l in 1:length(state)
        lbin=digits(l-1, base = 2, pad = N)
        abin=lbin[subsys] #finding the state of A within l
        bbin=PopMul(lbin,subsys) #finding the state of B within l
        prod[BinDec(abin)+1,BinDec(bbin)+1]=state[l]
    end
    redDM = zeros(Float64,2^Na,2^Na) #initializes the reduced density matrix
    for i in 1:2^Na
        for j in 1:2^Na
            redDM[i,j]=dot(prod[i,:],prod[j,:])
        end
    end
    redDM
    
end

function EntEntr(state,subsys) #determines the entanglement entropy
    
    redDM = RedDens(state,subsys)
    eigv = eigvals(redDM)
    entropy = 0
    for w in eigv
        if w > 1e-20
            entropy -= w*log(w)
        end
    end
    entropy
end
;