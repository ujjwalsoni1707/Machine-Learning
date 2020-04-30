t=int(input())
while(t>0):
    a=input().split(" ")
    b=a[0]
    c=a[1]
    d=len(b)-len(c)
    st=" "
    if len(b)>len(c):
        for i in range(len(c)-1,-1):
            st=str((int(b[i+d])+int(c[i]))%10)+st
        for j in range(len(b)-len(c)-1,-1):
            st=b[j]+st
    print(st)
    t=t-1