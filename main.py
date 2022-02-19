
def create_dataset(X,y):
    dataset=[]
    for h in range(10):
        new_list=[]
        for m in range(1,11):
            for l in range(99):
                a1=np.where(y==h)[0]
                a2=np.where(y!=h)[0]
                dataset.append(np.array([X[a1[0]],X[a1[m]],X[a2[l]]]))
    return dataset


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255

g=[]
z=[]
for t in range(10):
    i=0
    for j in range(50000):
        if t==y_train[j]:
            g.append(x_train[j])
            z.append(t)
            i+=1
        else:
            continue
        if i>=11:
            break
g1=[]
z1=[]
for t in range(10):
    i=0
    for j in range(10000):
        if t==y_test[j]:
            g1.append(x_test[j])
            z1.append(t)
            i+=1
        else:
            continue
        if i>=11:
            break

z=np.array(z)
z1=np.array(z1)
dataset=np.array(create_dataset(g,z)).reshape([-1,3,1,28,28,1])
test_data=np.array(create_dataset(g1,z1)).reshape([-1,3,1,28,28,1])
"""
print(dataset.shape)
dataset=dataset.tolist()
dataset=[[np.array(dataset[h][u]) for u in range(3)] for h in range(9900)]
"""
inp=tf.keras.Input(shape=(28,28,1))
inp1=Conv(64,(3,3),activation="relu")(inp)#,dilation_rate=2,model.output
x=Conv(64,(3,3),activation="relu")(inp1)#,dilation_rate=2
x=tf.keras.layers.MaxPooling2D((2,2))(x)
x=Conv(128,(3,3),activation="relu")(x)#,dilation_rate=2
x=Conv(128,(3,3),activation="relu")(x)#,dilation_rate=2
x=tf.keras.layers.Flatten()(x)
x=Dense(128)(x)
model8=tf.keras.Model(inputs=inp,outputs=x)#resnet.input
ground_layer=tf.keras.Input(shape=(28,28,1),name="ground")
pos_layer=tf.keras.Input(shape=(28,28,1),name="pos")
neg_layer=tf.keras.Input(shape=(28,28,1),name="neg")

class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DistanceLayer,self).__init__()
    def call(self,ground,pos,neg):
        ap = tf.reduce_sum(tf.square(ground - pos),-1)
        an = tf.reduce_sum(tf.square(ground - neg),-1)
        return ap, an
#model1=tf.keras.Model(inp,x)
distances=DistanceLayer()(model8(ground_layer),
                        model8(pos_layer),
                        model8(neg_layer))
model1=tf.keras.Model(inputs=[ground_layer,pos_layer,neg_layer],outputs=distances)
print(dataset.shape)
dataset=dataset.tolist()
dataset=[[np.array(dataset[h][u]) for u in range(3)] for h in range(9900)]
dataset=test_data.tolist()
test_data=[[np.array(test_data[h][u]) for u in range(3)] for h in range(9900)]
l=np.array(dataset[0])


class SiameseNN():
    def __init__(self,mdl,lr,m=0.2):

       # super(SiameseNN,self).__init__()
        self.model=mdl
        self.lr=lr
        self.m=m

    def distance(self,ground,pos,neg):
        ap = tf.reduce_sum(tf.square(ground - pos),-1)
        an = tf.reduce_sum(tf.square(ground - neg),-1)
        return ap, an

    def model_preparing(self,X):
        a1=X[0]
        a2=X[1]
        a3=X[2]
        return self.model(a1),self.model(a2),self.model(a3)
    def triplet_loss(self,data,m):
        #ap,an=self.model(ground,true,fake)
        ap,an=self.model(data)
        return tf.maximum(ap-an+m,0)
    def compute_grad(self,data,m):#ap,an,
        with tf.GradientTape() as tape:
            all_loss=self.triplet_loss(data,m)
        return tape.gradient(all_loss,self.model.trainable_weights),all_loss
    def train_step(self,X,epochs):
        for nm in range(epochs):
            opt=tf.keras.optimizers.Adam(learning_rate=self.lr)
            gradient,loss=self.compute_grad(X[nm],self.m)

            print("Epochs:" + str(nm) + "Loss:" + str(loss))
            #print(self.model.trainable_weights)
            if loss<=0:
                continue
            opt.apply_gradients(zip(gradient,self.model.trainable_weights))
    def test(self,X):
        ap,an=self.model(X)
        return ap,an

    def call(self,inputs):
        return self.model(inputs)
model1.load_weights("chpt")
model6=SiameseNN(model1,lr=0.001,m=5000000)

#model6.train_step(dataset,epochs=9900)
print(model6.test(test_data))
model1.save_weights("chpt")
