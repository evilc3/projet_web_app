import streamlit as st
import pandas as pd 
import numpy as np 
from string import punctuation
import pickle as pk



#preprocessing libraries
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re

import nltk
nltk.download('stopwords')

list_rc_sub = [['raw material','Excipent','Qualitative for.','Quantative for.','amount','appearance','storage condition','batchsize','Stability','contamination','expiry date'],
               ['Breakdown/failure','maintenance','cleaning','calibrating','qualification'],
               ['process control','Analysis','Sampling'],
               ['Procedure','batchrecord','spec. Als.','procedure  als.','Product Spec.','raw material spec.','spec. other'],
               ['air treatment','environment','water'],
               ['training','unknown']]

list_rc_cat = ['material/product','machine/apparatus/room','measurment','method_procedure/process','environment','human']


extended = ["a","a's","able","about","above","according","accordingly","across","actually","after",
                                "afterwards","again","against","ain't","all","allow","allows","almost","alone","along",
                                "already","also","although","always","am","among","amongst","an","and","another","any",
                                "anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart",
                                "appear","appreciate","appropriate","are","aren't","around","as","aside",
                                "ask","asking","associated","at","available","away","awfully","b","be",
                                "became","because","become","becomes","becoming","been","before","beforehand",
                                "behind","being","believe","below","beside","besides","best","better","between","beyond",
                                "both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly",
                                "changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing",
                                "contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't",
                                "different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either",
                                "else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere",
                                "ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former",
                                "formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going",
                                "gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's",
                                "hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his",
                                "hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc",
                                "indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its",
                                "itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less",
                                "lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean",
                                "meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly",
                                "necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally",
                                "not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only",
                                "onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular",
                                "particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r",
                                "rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","'s","said","same",
                                "saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible",
                                "sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow",
                                "someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub",
                                "such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their",
                                "theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these",
                                "they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through",
                                "throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u",
                                "un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp",
                                "v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome",
                                "well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby",
                                "wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will",
                                "willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're",
                                "you've","your","yours","yourself","yourselves","z","zero","html","ol"]
word_punct = set(stopwords.words('english')).union(punctuation).union(extended)

def decode_cat(ops):
    
    cat_ops = []    
    for k in ops:
        cat = [] 
        for y in k:
        
            if y.strip() == 'unknown':
                cat.append(y)
                break
        
            for i,j in enumerate(list_rc_sub):
            
                if y.strip() in j:
                    cat.append(list_rc_cat[i])
                    break
                
        cat_ops.append(cat)
    return cat_ops

def preprocessing(x):
    
    input = re.sub('[!@#$%^&*()\n_:><?\-.{}|+-,;""``~`—]|[0-9]|/|=|\[\]|\[\[\]\]',' ',x)
    input = re.sub('[“’\']','',input)  
    tmp = " "
    
    for i in word_tokenize(input):
        
        if i not in word_punct:
            tmp += i.lower() + " ";
    
    return tmp    
    
    


#app heading
st.title("         Error  App ")


# """ 
# how to use section...
# """


if st.checkbox("How to Use?"):
    st.write("The input is extracted from the Deviation Description and Product Names from the Deviation from.***See the example***")
    st.write("## INPUT")
    
    st.image("input_red.jpg",width = 900)
    st.write("### Extracted input")
    st.write("***Glycopyranium Bromide :: (Product name ) and***")
    st.write("***Deviation Description :: An unknown impurity is detected that 0.2%....***")
    st.write("this is entered in the **Input Area below and press the predict button**")

    st.write("## Output")

    st.image("out.png",width = 1000)

    st.write("**Understanding  output**")
    st.write("root cause found in *** 1 category***")  
    st.write("Category:: method_procedure/process") 
    st.write("Root cause:: Procedure 94.78755515965329") 
    st.write(" **The output matches with the root cause detected in the deviation form with *** 94.78%*** accuracy.**")
    st.image("true_output.png",width = 900)




st.write("# Input Area")
text_input = st.text_area(" ","Enter input here...")

processed_text = preprocessing(text_input)

#preporcessing 
clf = pk.load(open("model.pkl","rb"))
multilabel_binarizer = pk.load(open("label.pkl","rb"))
vec = pk.load(open("vect.pkl","rb"))



if st.button("Predict"):

    st.write("## ** Prediction **")

    #1 get the text input and apply preprocessing 
    
    data = vec.transform([processed_text])

    ops = clf.predict(data)
    labels = multilabel_binarizer.inverse_transform(ops)
    ops_prob = clf.predict_proba(data) * ops
    labels_prob = multilabel_binarizer.classes_[ops_prob[0]>0]
    
    

    ops_list = ops_prob[ops_prob != 0]
    

    cat = decode_cat(labels)
    
    if len(cat[0]) > 0:
        
        st.write(f'PROBABLE ROOT CAUSE DETECTED IN ** {len(cat[0])} categories **')
        categories = ""
        for i in cat[0][:]:
            categories += i + ","

        st.write(f' ## Categories:\n{categories}')

        st.write(f"## Root Causes")
        for i,j in enumerate(labels[0]):
            st.write("### "+j.strip(),ops_list[i] * 100)
    else:
        st.write("no root cause deteced please enter valid input")    
    

st.write('### Select/(copy paste) sample input form the table')



data = pd.read_csv("dataset.csv" , nrows = 20)


def formater(x):

    if x == '[]':
        return 'unknown'
    else:
        return re.sub('\[|\]|\'|','',x)
    # ls = []
    # for i in x[1:-1].split(','):
    #     ls.append(re.sub('\'','',i))  

    # return ls    


input_col = data['Pname'] + data['Desc.']

output_col = data['RC'].apply(formater)


dataset = pd.DataFrame(data = {"Input":input_col,"Root Causes":output_col})
    

st.table(dataset)


