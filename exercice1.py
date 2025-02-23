#ex1.1
import random
import numpy as np
import time
import matplotlib.pyplot as plt

for i in range(10):
    if i <= 5 :
        print('X'*i)
    else :
        print('X'*(10-i))

#ex1.2
input_str = "n45as29@#8ss6"
result = sum(int(char) for char in input_str if char.isdigit())
print(result)

#ex1.3
n=31
binary = []
while n>2:
    binary.insert(1,int(n%2))
    n=n/2
binary.insert(1,int(n%2))
print(binary[::-1])

#ex1.4
def fibonacci(upper_threshold: int) -> list:
    l = [0]
    n = 0
    m = 1
    while n+m<upper_threshold:
        l.append(n+m)
        k = n+m
        n = m
        m = k
    return l
#print(fibonacci(10))

#ex1.5
def rock_paper_scissors(n: int) -> str:
    user_score = 0
    computer_score = 0
    count = 0
    l = ['rock','paper','scissor']
    n = random.choice(l)
    while count < n:
        user_input = input("Enter your movement").strip().lower()
        if user_input == n:
            computer_score = computer_score + 1
            user_score = user_score + 1
        elif user_input == 'scissor' and n == 'rock':
            computer_score = computer_score + 1
        elif user_input == 'rock' and n == 'paper':
            computer_score = computer_score + 1
        elif user_input == 'paper' and n == 'scissor':
            computer_score=computer_score+1
        else:
            user_score=user_score+1
        count=+count
    if user_score<computer_score:
        return'You lose'
    elif user_score == computer_score:
        return'It is a tie'
    else:
        return'You win'
#ex2.1
def create_array_nxn(n: int) -> np.ndarray:
    return np.arange(n ** 2 - 1, -1, -1).reshape(n, n)
#print(create_array_nxn(2))
#test = create_array_nxn(2)
def apply_threshold_loop(m:np.ndarray, n: int)->np.ndarray:
    for idx, val in np.ndenumerate(m):
        if val < n:
            m[idx] = 0
    return m
 
#print(apply_threshold_loop(test,5))
def apply_threshold_vectorized(arr: np.ndarray, threshold: int) -> np.ndarray:
    m = np.where(arr < n, 0, arr)
    return m
#print(apply_threshold_vectorized(test,5))

def compare_performance(n: int, threshold: int) -> None:
    p = create_array_nxn(n)
    start1 = time.time()
    apply_threshold_loop(p,n)
    end1 = time.time()
    start2 = time.time()
    apply_threshold_vectorized(p,threshold)
    end2 = time.time()
    print('loop : ',end1-start1,'vectorized : ',end2-start2)

#print(compare_performance(1000,1000))

#ex2.2
#def show_in_digi(input_integer: int) -> None:
import numpy as np
import matplotlib.pyplot as plt

# Dictionnaire des chiffres en format matriciel
numbs = {
    "1": np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]),
    "2": np.array([[1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]]),
    "3": np.array([[1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
    "4": np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]),
    "5": np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
    "6": np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]]),
    "7": np.array([[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]),
    "8": np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]]),
    "9": np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
    "0": np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]]),
}


def display_number(number: int):
    """ Affiche un nombre entier sous forme de Digi Display """
    # Convertir le nombre en chaîne pour accéder aux chiffres
    str_number = str(number)

    # Liste des matrices correspondantes aux chiffres du nombre
    matrices = [numbs[digit] for digit in str_number]

    # Concaténer les matrices horizontalement pour former le nombre complet
    display_matrix = np.concatenate(matrices, axis=1)  # axis=1 pour concaténer horizontalement

    # Afficher avec Matplotlib
    plt.imshow(display_matrix, cmap="gray_r")  # "gray_r" pour que 1 soit noir et 0 blanc
    plt.axis("off")  # Supprime les axes
    plt.show()


# Exemple d'affichage
display_number(12345)
