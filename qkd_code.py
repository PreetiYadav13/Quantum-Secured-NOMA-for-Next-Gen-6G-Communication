import qiskit
# Other useful packages
import math
import matplotlib.pyplot as plt
import numpy as np

# Import Qiskit
from qiskit.providers.basic_provider import BasicProvider
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile

# Super secret message
mes = 'hello world'
print('Your super secret message: ', mes)

# Initial size of key
n = len(mes) * 5

# Break up message into smaller parts if length > 10
nlist = [10] * (n // 10) + ([n % 10] if n % 10 != 0 else [])

print('Initial key length:', n)

def randomStringGen(string_length):
    """ Generate a random quantum string of given length. """
    output = ''
    
    # Start up your quantum circuit information
    backend = BasicProvider().get_backend('basic_simulator') 

    # Run circuit in batches of 10 qubits for fastest results
    n = string_length
    temp_n = 10  # Batch size
    for _ in range(math.ceil(n / temp_n)):
        q = QuantumRegister(temp_n, name='q')
        c = ClassicalRegister(temp_n, name='c')
        rs = QuantumCircuit(q, c, name='rs')
        
        # Create temp_n number of qubits in superpositions
        for i in range(temp_n):
            rs.h(q[i])  # Hadamard gate for superposition
            rs.measure(q[i], c[i])

        # Execute the circuit
        new_circuit = transpile(rs, backend)
        job = backend.run(new_circuit)  
        result = job.result()  
        result_key = list(result.get_counts(rs).keys())[0]  # Extract binary string
        
        output += result_key
    
    return output[:n]

key = randomStringGen(n)
print('Initial key: ',key)

# Generate random rotation strings for Alice and Bob
Alice_rotate = randomStringGen(n)
Bob_rotate = randomStringGen(n)
print("Alice's rotation string:", Alice_rotate)
print("Bob's rotation string:  ", Bob_rotate)

# Start up your quantum program
backend = BasicProvider().get_backend('basic_simulator') 
Bob_result = ''

for ind, l in enumerate(nlist):
    # Define temp variables used in breaking up quantum program
    key_temp = key[10 * ind:10 * ind + l] if l < 10 else key[l * ind:l * (ind + 1)]
    Ar_temp = Alice_rotate[10 * ind:10 * ind + l] if l < 10 else Alice_rotate[l * ind:l * (ind + 1)]
    Br_temp = Bob_rotate[10 * ind:10 * ind + l] if l < 10 else Bob_rotate[l * ind:l * (ind + 1)]

    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    send_over = QuantumCircuit(q, c, name='send_over')
    
    # Prepare qubits based on the key and apply Hadamard gates
    for i, j, k, n in zip(key_temp, Ar_temp, Br_temp, range(len(key_temp))):
        i, j, k = int(i), int(j), int(k)
        if i > 0:
            send_over.x(q[n])  # Apply X gate if key bit is 1
        if j > 0:
            send_over.h(q[n])  # Alice's Hadamard rotation
        if k > 0:
            send_over.h(q[n])  # Bob's Hadamard rotation
        send_over.measure(q[n], c[n])

    # Execute circuit
    new_circuit = transpile(send_over, backend)
    job = backend.run(new_circuit)
    result = job.result()
    result_key = list(result.get_counts(send_over).keys())[0][::-1]  # Reverse for correct order
    Bob_result += result_key
    
print("Bob's results: ", Bob_result)

def makeKey(rotation1, rotation2, results):
    """ Generate final key by matching Alice and Bob's bases. """
    return ''.join(results[i] for i in range(len(rotation1)) if rotation1[i] == rotation2[i])

# Generate the final keys
Akey = makeKey(Bob_rotate, Alice_rotate, key)
Bkey = makeKey(Bob_rotate, Alice_rotate, Bob_result)

print("Alice's key:", Akey)
print("Bob's key:  ", Bkey)

# Make key same length as message
shortened_Akey = Akey[:len(mes)]
encoded_m = ''

# Encrypt message using key
for m, k in zip(mes, shortened_Akey):
    encoded_c = chr((ord(m) + 2 * ord(k)) % 256)
    encoded_m += encoded_c

print('Encoded message:  ', encoded_m)

# Make key same length as message
shortened_Bkey = Bkey[:len(mes)]
decoded_m = ''

# Decrypt message using key
for m, k in zip(encoded_m, shortened_Bkey):
    decoded_c = chr((ord(m) - 2 * ord(k)) % 256)
    decoded_m += decoded_c

print('Recovered message:', decoded_m)



backend = BasicProvider().get_backend('basic_simulator')  
shots = 1  
circuits = ['Eve']

Eve_result = ''
for ind, l in enumerate(nlist):
    if l < 10:
        key_temp = key[10 * ind:10 * ind + l]
        Ar_temp = Alice_rotate[10 * ind:10 * ind + l]
    else:
        key_temp = key[l * ind:l * (ind + 1)]
        Ar_temp = Alice_rotate[l * ind:l * (ind + 1)]
    
    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    Eve = QuantumCircuit(q, c, name='Eve')

    for i, j, n in zip(key_temp, Ar_temp, range(len(key_temp))):
        if int(i) > 0:
            Eve.x(q[n])
        if int(j) > 0:
            Eve.h(q[n])
        Eve.measure(q[n], c[n])

    circuit = transpile(Eve, backend)
    new_job = backend.run(circuit)
    result_eve = new_job.result()
    counts_eve = result_eve.get_counts()
    
    if counts_eve:
        result_key_eve = list(counts_eve.keys())
        Eve_result += result_key_eve[0][::-1]  # Reverse string to match expected output

print("Eve's results:", Eve_result)


#start up your quantum program
backend = BasicProvider().get_backend('basic_simulator')  
shots = 1
circuits = ['Eve2']

Bob_badresult = ''
for ind,l in enumerate(nlist):
    #define temp variables used in breaking up quantum program if message length > 10
    if l < 10:
        key_temp = key[10*ind:10*ind+l]
        Eve_temp = Eve_result[10*ind:10*ind+l]
        Br_temp = Bob_rotate[10*ind:10*ind+l]
    else:
        key_temp = key[l*ind:l*(ind+1)]
        Eve_temp = Eve_result[l*ind:l*(ind+1)]
        Br_temp = Bob_rotate[l*ind:l*(ind+1)]
    
    #start up the rest of your quantum circuit information
    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    Eve2 = QuantumCircuit(q , c, name='Eve2')
    
    #prepare qubits
    for i,j,n in zip(Eve_temp,Br_temp,range(0,len(key_temp))):
        i = int(i)
        j = int(j)
        if i > 0:
            Eve2.x(q[n])
        if j > 0:
            Eve2.h(q[n])
        Eve2.measure(q[n],c[n])
    
    #execute
    circuit = transpile(Eve2, backend, shots)
    new_job = backend.run(circuit)
    result_eve = new_job.result()
    counts_eve = result_eve.get_counts()
    result_key_eve = list(result_eve.get_counts().keys())
    Bob_badresult += result_key_eve[0][::-1]
    
print("Bob's previous results (w/o Eve):",Bob_result)
print("Bob's results from Eve:\t\t ",Bob_badresult)



 #make keys for Alice and Bob
Akey = makeKey(Bob_rotate,Alice_rotate,key)
Bkey = makeKey(Bob_rotate,Alice_rotate,Bob_badresult)
print("Alice's key:   ",Akey)
print("Bob's key:     ",Bkey)

check_key = randomStringGen(len(Akey))
print('spots to check:',check_key)



#find which values in rotation string were used to make the key
Alice_keyrotate = makeKey(Bob_rotate,Alice_rotate,Alice_rotate)
Bob_keyrotate = makeKey(Bob_rotate,Alice_rotate,Bob_rotate)

# Detect Eve's interference
#extract a subset of Alice's key
sub_Akey = ''
sub_Arotate = ''
count = 0
for i,j in zip(Alice_rotate,Akey):
    if int(check_key[count]) == 1:
        sub_Akey += Akey[count]
        sub_Arotate += Alice_keyrotate[count]
    count += 1

#extract a subset of Bob's key
sub_Bkey = ''
sub_Brotate = ''
count = 0
for i,j in zip(Bob_rotate,Bkey):
    if int(check_key[count]) == 1:
        sub_Bkey += Bkey[count]
        sub_Brotate += Bob_keyrotate[count]
    count += 1
print("subset of Alice's key:",sub_Akey)
print("subset of Bob's key:  ",sub_Bkey)

#compare Alice and Bob's key subsets
secure = True
for i,j in zip(sub_Akey,sub_Bkey):
    if i == j:
        secure = True
    else:
        secure = False
        break;
if not secure:
    print('Eve detected!')
else:
    print('Eve escaped detection!')

#sub_Akey and sub_Bkey are public knowledge now, so we remove them from Akey and Bkey
if secure:
    new_Akey = ''
    new_Bkey = ''
    for index,i in enumerate(check_key):
        if int(i) == 0:
            new_Akey += Akey[index]
            new_Bkey += Bkey[index]
    print('new A and B keys: ',new_Akey,new_Bkey)
    if(len(mes)>len(new_Akey)):
        print('Your new key is not long enough.')

# Compute Quantum Bit Error Rate (QBER)
if len(Akey) > 0 and len(Bkey) > 0:
    mismatches = sum(1 for a, b in zip(Akey, Bkey) if a != b)
    QBER = mismatches / len(Akey)
    print(f"Quantum Bit Error Rate (QBER): {QBER:.4f}")
else:
    print("Not enough key bits to compute QBER.")



x = np.arange(0., 50.0)
y = 1-(3/4)**x
plt.plot(y)
plt.title('Probablity of detecting Eve')
plt.xlabel('# of key bits compared')
plt.ylabel('Probablity of detecting Eve')
plt.show()
