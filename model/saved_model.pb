??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.0-beta12v2.0.0-beta0-16-g1d91213fe78ޚ
~
conv2d/kernelVarHandleOp*
dtype0*
shared_nameconv2d/kernel*
shape:*
_output_shapes
: 
?
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:
n
conv2d/biasVarHandleOp*
shape:*
_output_shapes
: *
dtype0*
shared_nameconv2d/bias
?
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0*
_class
loc:@conv2d/bias
?
conv2d_1/kernelVarHandleOp*
shape:*
_output_shapes
: * 
shared_nameconv2d_1/kernel*
dtype0
?
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:*"
_class
loc:@conv2d_1/kernel
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
shape:*
dtype0*
shared_nameconv2d_1/bias
?
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
v
dense/kernelVarHandleOp*
shape:
??*
dtype0*
_output_shapes
: *
shared_namedense/kernel
?
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0* 
_output_shapes
:
??
l

dense/biasVarHandleOp*
dtype0*
shared_name
dense/bias*
shape:*
_output_shapes
: 
?
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
}
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
_class
loc:@Adam/iter*
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
shared_nameAdam/beta_1*
dtype0*
shape: 
?
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0*
_class
loc:@Adam/beta_1
j
Adam/beta_2VarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nameAdam/beta_2
?
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
_output_shapes
: *
dtype0*
shared_name
Adam/decay
?
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_class
loc:@Adam/decay*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*#
shared_nameAdam/learning_rate*
shape: *
_output_shapes
: *
dtype0
?
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*%
_class
loc:@Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
shared_nametotal*
shape: *
_output_shapes
: 
q
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0*
_class

loc:@total
^
countVarHandleOp*
shape: *
_output_shapes
: *
dtype0*
shared_namecount
q
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_class

loc:@count*
_output_shapes
: 
t
true_positivesVarHandleOp*
shape:*
dtype0*
shared_nametrue_positives*
_output_shapes
: 
?
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*!
_class
loc:@true_positives*
dtype0*
_output_shapes
:
v
false_positivesVarHandleOp* 
shared_namefalse_positives*
shape:*
_output_shapes
: *
dtype0
?
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*"
_class
loc:@false_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
dtype0*!
shared_nametrue_positives_1*
_output_shapes
: *
shape:
?
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*#
_class
loc:@true_positives_1*
dtype0*
_output_shapes
:
v
false_negativesVarHandleOp* 
shared_namefalse_negatives*
_output_shapes
: *
shape:*
dtype0
?
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
dtype0*
_output_shapes
:*"
_class
loc:@false_negatives
y
true_positives_2VarHandleOp*
shape:?*
_output_shapes
: *!
shared_nametrue_positives_2*
dtype0
?
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:?*#
_class
loc:@true_positives_2*
dtype0
u
true_negativesVarHandleOp*
dtype0*
_output_shapes
: *
shared_nametrue_negatives*
shape:?
?
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*!
_class
loc:@true_negatives*
_output_shapes	
:?*
dtype0
{
false_positives_1VarHandleOp*"
shared_namefalse_positives_1*
_output_shapes
: *
dtype0*
shape:?
?
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:?*
dtype0*$
_class
loc:@false_positives_1
{
false_negatives_1VarHandleOp*
dtype0*
shape:?*"
shared_namefalse_negatives_1*
_output_shapes
: 
?
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:?*
dtype0*$
_class
loc:@false_negatives_1
?
Adam/conv2d/kernel/mVarHandleOp*
shape:*%
shared_nameAdam/conv2d/kernel/m*
dtype0*
_output_shapes
: 
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*'
_class
loc:@Adam/conv2d/kernel/m*
dtype0*&
_output_shapes
:
|
Adam/conv2d/bias/mVarHandleOp*
dtype0*#
shared_nameAdam/conv2d/bias/m*
shape:*
_output_shapes
: 
?
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*%
_class
loc:@Adam/conv2d/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0*)
_class
loc:@Adam/conv2d_1/kernel/m
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_1/bias/m*
shape:*
dtype0
?
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
dtype0*
_output_shapes
:*'
_class
loc:@Adam/conv2d_1/bias/m
?
Adam/dense/kernel/mVarHandleOp*
shape:
??*$
shared_nameAdam/dense/kernel/m*
_output_shapes
: *
dtype0
?
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
dtype0* 
_output_shapes
:
??*&
_class
loc:@Adam/dense/kernel/m
z
Adam/dense/bias/mVarHandleOp*
shape:*
dtype0*"
shared_nameAdam/dense/bias/m*
_output_shapes
: 
?
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*$
_class
loc:@Adam/dense/bias/m*
_output_shapes
:
?
Adam/conv2d/kernel/vVarHandleOp*
shape:*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/v*
dtype0
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*
dtype0*&
_output_shapes
:*'
_class
loc:@Adam/conv2d/kernel/v
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
?
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*%
_class
loc:@Adam/conv2d/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*'
shared_nameAdam/conv2d_1/kernel/v*
shape:
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0*)
_class
loc:@Adam/conv2d_1/kernel/v
?
Adam/conv2d_1/bias/vVarHandleOp*%
shared_nameAdam/conv2d_1/bias/v*
shape:*
_output_shapes
: *
dtype0
?
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
:*'
_class
loc:@Adam/conv2d_1/bias/v
?
Adam/dense/kernel/vVarHandleOp*$
shared_nameAdam/dense/kernel/v*
dtype0*
_output_shapes
: *
shape:
??
?
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*&
_class
loc:@Adam/dense/kernel/v* 
_output_shapes
:
??
z
Adam/dense/bias/vVarHandleOp*
shape:*
_output_shapes
: *"
shared_nameAdam/dense/bias/v*
dtype0
?
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*$
_class
loc:@Adam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?@
ConstConst"/device:CPU:0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
{
_callable_losses
_eager_losses
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
bias
_callable_losses
_eager_losses
regularization_losses
trainable_variables
 	variables
!	keras_api
{
"_callable_losses
#_eager_losses
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?

(kernel
)bias
*_callable_losses
+_eager_losses
,regularization_losses
-trainable_variables
.	variables
/	keras_api
{
0_callable_losses
1_eager_losses
2regularization_losses
3trainable_variables
4	variables
5	keras_api
{
6_callable_losses
7_eager_losses
8regularization_losses
9trainable_variables
:	variables
;	keras_api
?

<kernel
=bias
>_callable_losses
?_eager_losses
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
{
D_callable_losses
E_eager_losses
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m?(m?)m?<m?=m?v?v?(v?)v?<v?=v?
 
*
0
1
(2
)3
<4
=5
*
0
1
(2
)3
<4
=5
y
regularization_losses
Onon_trainable_variables

Players
	variables
trainable_variables
Qmetrics
 
 
 
 
y
regularization_losses
Rnon_trainable_variables

Slayers
trainable_variables
	variables
Tmetrics
 
 
 
 
 
y
regularization_losses
Unon_trainable_variables

Vlayers
trainable_variables
	variables
Wmetrics
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1

0
1
y
regularization_losses
Xnon_trainable_variables

Ylayers
trainable_variables
 	variables
Zmetrics
 
 
 
 
 
y
$regularization_losses
[non_trainable_variables

\layers
%trainable_variables
&	variables
]metrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

(0
)1

(0
)1
y
,regularization_losses
^non_trainable_variables

_layers
-trainable_variables
.	variables
`metrics
 
 
 
 
 
y
2regularization_losses
anon_trainable_variables

blayers
3trainable_variables
4	variables
cmetrics
 
 
 
 
 
y
8regularization_losses
dnon_trainable_variables

elayers
9trainable_variables
:	variables
fmetrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

<0
=1

<0
=1
y
@regularization_losses
gnon_trainable_variables

hlayers
Atrainable_variables
B	variables
imetrics
 
 
 
 
 
y
Fregularization_losses
jnon_trainable_variables

klayers
Gtrainable_variables
H	variables
lmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
	7

m0
n1
o2
p3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
	qtotal
	rcount
s
_fn_kwargs
t_updates
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?
y
thresholds
ztrue_positives
{false_positives
|_updates
}regularization_losses
~trainable_variables
	variables
?	keras_api
?
?
thresholds
?true_positives
?false_negatives
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?
thresholds
?_layers
?true_positives
?true_negatives
?false_positives
?false_negatives
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

q0
r1
|
uregularization_losses
?non_trainable_variables
?layers
vtrainable_variables
w	variables
?metrics
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

z0
{1
|
}regularization_losses
?non_trainable_variables
?layers
~trainable_variables
	variables
?metrics
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1

?regularization_losses
?non_trainable_variables
?layers
?trainable_variables
?	variables
?metrics
 

?0
?1
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?0
?1
?2
?3

?regularization_losses
?non_trainable_variables
?layers
?trainable_variables
?	variables
?metrics

q0
r1
 
 

z0
{1
 
 

?0
?1
 
 
 
?0
?1
?2
?3
 
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
?
$serving_default_zero_padding2d_inputPlaceholder*
dtype0*$
shape:?????????dd*/
_output_shapes
:?????????dd
?
StatefulPartitionedCallStatefulPartitionedCall$serving_default_zero_padding2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/bias*.
config_proto

GPU 

CPU(2J 8*'
_output_shapes
:?????????*,
f'R%
#__inference_signature_wrapper_95278*
Tin
	2*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*.
config_proto

GPU 

CPU(2J 8*.
Tin'
%2#	*'
f"R 
__inference__traced_save_95403*
Tout
2*,
_gradient_op_typePartitionedCall-95404*
_output_shapes
: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*.
config_proto

GPU 

CPU(2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-95516**
f%R#
!__inference__traced_restore_95515*
_output_shapes
: *-
Tin&
$2"??
?
?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95194
zero_padding2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallzero_padding2d_input*,
_gradient_op_typePartitionedCall-95000*/
_output_shapes
:?????????hh*
Tout
2*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994*
Tin
2*.
config_proto

GPU 

CPU(2J 8?
conv2d/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-95023*
Tin
2*/
_output_shapes
:?????????dd*
Tout
2*.
config_proto

GPU 

CPU(2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_95017?
 zero_padding2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tout
2*T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037*/
_output_shapes
:?????????hh*
Tin
2*,
_gradient_op_typePartitionedCall-95043*.
config_proto

GPU 

CPU(2J 8?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*/
_output_shapes
:?????????dd*,
_gradient_op_typePartitionedCall-95066*
Tin
2*
Tout
2*.
config_proto

GPU 

CPU(2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95085*
Tout
2*.
config_proto

GPU 

CPU(2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079*/
_output_shapes
:?????????22*
Tin
2?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*)
_output_shapes
:???????????*.
config_proto

GPU 

CPU(2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-95121*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95115*
Tin
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_95138*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95144*
Tout
2*
Tin
2*'
_output_shapes
:??????????
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_95160*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-95166*
Tin
2*.
config_proto

GPU 

CPU(2J 8?
IdentityIdentity#activation/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:4 0
.
_user_specified_namezero_padding2d_input: : : : : : 
?A
?
__inference__traced_save_95403
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_8dc75556e47b4a4e8e0cb33dc9653cd4/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:!*?
value?B?!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::
??:: : : : : : : :::::?:?:?:?:::::
??::::::
??:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints: : : : : : : : : : : : : : : : : : : :  :! :" :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_95138

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
3__inference_AllOneApiClassifier_layer_call_fn_95257
zero_padding2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
config_proto

GPU 

CPU(2J 8*'
_output_shapes
:?????????*
Tin
	2*,
_gradient_op_typePartitionedCall-95248*
Tout
2*W
fRRP
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95247?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::22
StatefulPartitionedCallStatefulPartitionedCall:4 0
.
_user_specified_namezero_padding2d_input: : : : : : 
?
F
*__inference_activation_layer_call_fn_95169

inputs
identity?
PartitionedCallPartitionedCallinputs*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_95160*'
_output_shapes
:?????????*.
config_proto

GPU 

CPU(2J 8*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-95166`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
e
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95215

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tout
2*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????hh*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994*,
_gradient_op_typePartitionedCall-95000*
Tin
2?
conv2d/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_95017*
Tin
2*/
_output_shapes
:?????????dd*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95023?
 zero_padding2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????hh*T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-95043?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-95066*
Tout
2*/
_output_shapes
:?????????dd*.
config_proto

GPU 

CPU(2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079*,
_gradient_op_typePartitionedCall-95085*
Tout
2*.
config_proto

GPU 

CPU(2J 8*
Tin
2*/
_output_shapes
:?????????22?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95121*)
_output_shapes
:???????????*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95115*
Tin
2*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*.
config_proto

GPU 

CPU(2J 8*
Tin
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_95138*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-95144*
Tout
2?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*.
config_proto

GPU 

CPU(2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_95160*
Tin
2*'
_output_shapes
:?????????*
Tout
2*,
_gradient_op_typePartitionedCall-95166?
IdentityIdentity#activation/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?~
?
!__inference__traced_restore_95515
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count&
"assignvariableop_13_true_positives'
#assignvariableop_14_false_positives(
$assignvariableop_15_true_positives_1'
#assignvariableop_16_false_negatives(
$assignvariableop_17_true_positives_2&
"assignvariableop_18_true_negatives)
%assignvariableop_19_false_positives_1)
%assignvariableop_20_false_negatives_1,
(assignvariableop_21_adam_conv2d_kernel_m*
&assignvariableop_22_adam_conv2d_bias_m.
*assignvariableop_23_adam_conv2d_1_kernel_m,
(assignvariableop_24_adam_conv2d_1_bias_m+
'assignvariableop_25_adam_dense_kernel_m)
%assignvariableop_26_adam_dense_bias_m,
(assignvariableop_27_adam_conv2d_kernel_v*
&assignvariableop_28_adam_conv2d_bias_v.
*assignvariableop_29_adam_conv2d_1_kernel_v,
(assignvariableop_30_adam_conv2d_1_bias_v+
'assignvariableop_31_adam_dense_kernel_v)
%assignvariableop_32_adam_dense_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:!?
RestoreV2/shape_and_slicesConst"/device:CPU:0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:!?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*/
dtypes%
#2!	*?
_output_shapes?
?:::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:|
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:~
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0}
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:{
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_2Identity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_false_positives_1Identity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_false_negatives_1Identity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_conv2d_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_conv2d_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_1_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_1_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_vIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_11: : : : : : : : : : : : : : : : :  :! :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : 
?
I
-__inference_max_pooling2d_layer_call_fn_95088

inputs
identity?
PartitionedCallPartitionedCallinputs*.
config_proto

GPU 

CPU(2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2*,
_gradient_op_typePartitionedCall-95085?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?	
?
3__inference_AllOneApiClassifier_layer_call_fn_95225
zero_padding2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*,
_gradient_op_typePartitionedCall-95216*W
fRRP
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95215*'
_output_shapes
:?????????*
Tout
2*.
config_proto

GPU 

CPU(2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :4 0
.
_user_specified_namezero_padding2d_input: : 
?
L
0__inference_zero_padding2d_1_layer_call_fn_95046

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037*,
_gradient_op_typePartitionedCall-95043*J
_output_shapes8
6:4????????????????????????????????????*.
config_proto

GPU 

CPU(2J 8*
Tout
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?:
?
 __inference__wrapped_model_94985
zero_padding2d_input=
9alloneapiclassifier_conv2d_conv2d_readvariableop_resource>
:alloneapiclassifier_conv2d_biasadd_readvariableop_resource?
;alloneapiclassifier_conv2d_1_conv2d_readvariableop_resource@
<alloneapiclassifier_conv2d_1_biasadd_readvariableop_resource<
8alloneapiclassifier_dense_matmul_readvariableop_resource=
9alloneapiclassifier_dense_biasadd_readvariableop_resource
identity??1AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOp?0AllOneApiClassifier/conv2d/Conv2D/ReadVariableOp?3AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOp?2AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOp?0AllOneApiClassifier/dense/BiasAdd/ReadVariableOp?/AllOneApiClassifier/dense/MatMul/ReadVariableOp?
/AllOneApiClassifier/zero_padding2d/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:?
&AllOneApiClassifier/zero_padding2d/PadPadzero_padding2d_input8AllOneApiClassifier/zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????hh?
0AllOneApiClassifier/conv2d/Conv2D/ReadVariableOpReadVariableOp9alloneapiclassifier_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0?
!AllOneApiClassifier/conv2d/Conv2DConv2D/AllOneApiClassifier/zero_padding2d/Pad:output:08AllOneApiClassifier/conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????dd*
T0*
strides
*
paddingVALID?
1AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOpReadVariableOp:alloneapiclassifier_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
"AllOneApiClassifier/conv2d/BiasAddBiasAdd*AllOneApiClassifier/conv2d/Conv2D:output:09AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????dd*
T0?
AllOneApiClassifier/conv2d/ReluRelu+AllOneApiClassifier/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????dd*
T0?
1AllOneApiClassifier/zero_padding2d_1/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:?
(AllOneApiClassifier/zero_padding2d_1/PadPad-AllOneApiClassifier/conv2d/Relu:activations:0:AllOneApiClassifier/zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????hh?
2AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;alloneapiclassifier_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:?
#AllOneApiClassifier/conv2d_1/Conv2DConv2D1AllOneApiClassifier/zero_padding2d_1/Pad:output:0:AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*
strides
*/
_output_shapes
:?????????dd?
3AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<alloneapiclassifier_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
$AllOneApiClassifier/conv2d_1/BiasAddBiasAdd,AllOneApiClassifier/conv2d_1/Conv2D:output:0;AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????dd*
T0?
!AllOneApiClassifier/conv2d_1/ReluRelu-AllOneApiClassifier/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????dd*
T0?
)AllOneApiClassifier/max_pooling2d/MaxPoolMaxPool/AllOneApiClassifier/conv2d_1/Relu:activations:0*
paddingVALID*
strides
*
ksize
*/
_output_shapes
:?????????22?
!AllOneApiClassifier/flatten/ShapeShape2AllOneApiClassifier/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:y
/AllOneApiClassifier/flatten/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0{
1AllOneApiClassifier/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:{
1AllOneApiClassifier/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)AllOneApiClassifier/flatten/strided_sliceStridedSlice*AllOneApiClassifier/flatten/Shape:output:08AllOneApiClassifier/flatten/strided_slice/stack:output:0:AllOneApiClassifier/flatten/strided_slice/stack_1:output:0:AllOneApiClassifier/flatten/strided_slice/stack_2:output:0*
T0*
_output_shapes
: *
Index0*
shrink_axis_maskv
+AllOneApiClassifier/flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: ?
)AllOneApiClassifier/flatten/Reshape/shapePack2AllOneApiClassifier/flatten/strided_slice:output:04AllOneApiClassifier/flatten/Reshape/shape/1:output:0*
_output_shapes
:*
T0*
N?
#AllOneApiClassifier/flatten/ReshapeReshape2AllOneApiClassifier/max_pooling2d/MaxPool:output:02AllOneApiClassifier/flatten/Reshape/shape:output:0*
T0*)
_output_shapes
:????????????
/AllOneApiClassifier/dense/MatMul/ReadVariableOpReadVariableOp8alloneapiclassifier_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
 AllOneApiClassifier/dense/MatMulMatMul,AllOneApiClassifier/flatten/Reshape:output:07AllOneApiClassifier/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0AllOneApiClassifier/dense/BiasAdd/ReadVariableOpReadVariableOp9alloneapiclassifier_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
!AllOneApiClassifier/dense/BiasAddBiasAdd*AllOneApiClassifier/dense/MatMul:product:08AllOneApiClassifier/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
&AllOneApiClassifier/activation/SoftmaxSoftmax*AllOneApiClassifier/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity0AllOneApiClassifier/activation/Softmax:softmax:02^AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOp1^AllOneApiClassifier/conv2d/Conv2D/ReadVariableOp4^AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOp3^AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOp1^AllOneApiClassifier/dense/BiasAdd/ReadVariableOp0^AllOneApiClassifier/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::2j
3AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOp3AllOneApiClassifier/conv2d_1/BiasAdd/ReadVariableOp2h
2AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOp2AllOneApiClassifier/conv2d_1/Conv2D/ReadVariableOp2b
/AllOneApiClassifier/dense/MatMul/ReadVariableOp/AllOneApiClassifier/dense/MatMul/ReadVariableOp2d
0AllOneApiClassifier/dense/BiasAdd/ReadVariableOp0AllOneApiClassifier/dense/BiasAdd/ReadVariableOp2d
0AllOneApiClassifier/conv2d/Conv2D/ReadVariableOp0AllOneApiClassifier/conv2d/Conv2D/ReadVariableOp2f
1AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOp1AllOneApiClassifier/conv2d/BiasAdd/ReadVariableOp: : : : : : :4 0
.
_user_specified_namezero_padding2d_input
?
?
(__inference_conv2d_1_layer_call_fn_95071

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060*A
_output_shapes/
-:+???????????????????????????*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95066*
Tin
2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
%__inference_dense_layer_call_fn_95149

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
config_proto

GPU 

CPU(2J 8*'
_output_shapes
:?????????*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_95138*,
_gradient_op_typePartitionedCall-95144*
Tin
2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
a
E__inference_activation_layer_call_and_return_conditional_losses_95160

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95174
zero_padding2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallzero_padding2d_input*
Tout
2*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994*,
_gradient_op_typePartitionedCall-95000*
Tin
2*/
_output_shapes
:?????????hh*.
config_proto

GPU 

CPU(2J 8?
conv2d/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:?????????dd*.
config_proto

GPU 

CPU(2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_95017*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-95023?
 zero_padding2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*.
config_proto

GPU 

CPU(2J 8*T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037*/
_output_shapes
:?????????hh*
Tout
2*,
_gradient_op_typePartitionedCall-95043?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????dd*
Tout
2*,
_gradient_op_typePartitionedCall-95066*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????22*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCall-95085*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95121*
Tin
2*)
_output_shapes
:???????????*.
config_proto

GPU 

CPU(2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95115*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*.
config_proto

GPU 

CPU(2J 8*'
_output_shapes
:?????????*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_95138*
Tout
2*,
_gradient_op_typePartitionedCall-95144?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95166*
Tout
2*
Tin
2*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_95160*'
_output_shapes
:?????????*.
config_proto

GPU 

CPU(2J 8?
IdentityIdentity#activation/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:4 0
.
_user_specified_namezero_padding2d_input: : : : : : 
?
g
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037

inputs
identity}
Pad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0~
PadPadinputsPad/paddings:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079

inputs
identity?
MaxPoolMaxPoolinputs*
paddingVALID*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95247

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinputs*/
_output_shapes
:?????????hh*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95000*
Tout
2*
Tin
2*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994?
conv2d/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_95017*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????dd*,
_gradient_op_typePartitionedCall-95023*
Tin
2?
 zero_padding2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95043*.
config_proto

GPU 

CPU(2J 8*/
_output_shapes
:?????????hh*
Tout
2*
Tin
2*T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060*,
_gradient_op_typePartitionedCall-95066*.
config_proto

GPU 

CPU(2J 8*
Tout
2*
Tin
2*/
_output_shapes
:?????????dd?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079*
Tin
2*,
_gradient_op_typePartitionedCall-95085*
Tout
2*/
_output_shapes
:?????????22*.
config_proto

GPU 

CPU(2J 8?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95121*
Tin
2*
Tout
2*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95115*)
_output_shapes
:???????????*.
config_proto

GPU 

CPU(2J 8?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-95144*
Tout
2*'
_output_shapes
:?????????*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_95138*
Tin
2*.
config_proto

GPU 

CPU(2J 8?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-95166*'
_output_shapes
:?????????*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_95160*.
config_proto

GPU 

CPU(2J 8*
Tout
2*
Tin
2?
IdentityIdentity#activation/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : : : : :& "
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_95278
zero_padding2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*,
_gradient_op_typePartitionedCall-95269*
Tout
2*
Tin
	2*)
f$R"
 __inference__wrapped_model_94985*.
config_proto

GPU 

CPU(2J 8*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????dd::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :4 0
.
_user_specified_namezero_padding2d_input: 
?
C
'__inference_flatten_layer_call_fn_95124

inputs
identity?
PartitionedCallPartitionedCallinputs*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95115*
Tout
2*.
config_proto

GPU 

CPU(2J 8*,
_gradient_op_typePartitionedCall-95121*)
_output_shapes
:???????????*
Tin
2b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*.
_input_shapes
:?????????22:& "
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_95017

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
strides
*
paddingVALID?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?	
^
B__inference_flatten_layer_call_and_return_conditional_losses_95115

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: Z
Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
_output_shapes
:*
N*
T0f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*)
_output_shapes
:???????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????22:& "
 
_user_specified_nameinputs
?
J
.__inference_zero_padding2d_layer_call_fn_95003

inputs
identity?
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4????????????????????????????????????*.
config_proto

GPU 

CPU(2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-95000*
Tout
2*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_95028

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*
Tin
2*.
config_proto

GPU 

CPU(2J 8*A
_output_shapes/
-:+???????????????????????????*,
_gradient_op_typePartitionedCall-95023*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_95017?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs"7L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
]
zero_padding2d_inputE
&serving_default_zero_padding2d_input:0?????????dd>

activation0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?R
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?O
_tf_keras_sequential?O{"class_name": "Sequential", "name": "AllOneApiClassifier", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "AllOneApiClassifier", "layers": [{"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "AllOneApiClassifier", "layers": [{"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [{"class_name": "BinaryAccuracy", "config": {"name": "acc", "dtype": "float32", "threshold": 0.5}}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593]}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 9.999999974752427e-07, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "zero_padding2d_input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100, 100, 3], "config": {"batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "sparse": false, "name": "zero_padding2d_input"}, "input_spec": null, "activity_regularizer": null}
?
_callable_losses
_eager_losses
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100, 100, 3], "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
?

kernel
bias
_callable_losses
_eager_losses
regularization_losses
trainable_variables
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "activity_regularizer": null}
?
"_callable_losses
#_eager_losses
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": [[2, 2], [2, 2]], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
?

(kernel
)bias
*_callable_losses
+_eager_losses
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 30}}}, "activity_regularizer": null}
?
0_callable_losses
1_eager_losses
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
?
6_callable_losses
7_eager_losses
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "activity_regularizer": null}
?

<kernel
=bias
>_callable_losses
?_eager_losses
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75000}}}, "activity_regularizer": null}
?
D_callable_losses
E_eager_losses
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}, "input_spec": null, "activity_regularizer": null}
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m?(m?)m?<m?=m?v?v?(v?)v?<v?=v?"
	optimizer
 "
trackable_list_wrapper
J
0
1
(2
)3
<4
=5"
trackable_list_wrapper
J
0
1
(2
)3
<4
=5"
trackable_list_wrapper
?
regularization_losses
Onon_trainable_variables

Players
	variables
trainable_variables
Qmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Rnon_trainable_variables

Slayers
trainable_variables
	variables
Tmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Unon_trainable_variables

Vlayers
trainable_variables
	variables
Wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Xnon_trainable_variables

Ylayers
trainable_variables
 	variables
Zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$regularization_losses
[non_trainable_variables

\layers
%trainable_variables
&	variables
]metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
,regularization_losses
^non_trainable_variables

_layers
-trainable_variables
.	variables
`metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
2regularization_losses
anon_trainable_variables

blayers
3trainable_variables
4	variables
cmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8regularization_losses
dnon_trainable_variables

elayers
9trainable_variables
:	variables
fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
@regularization_losses
gnon_trainable_variables

hlayers
Atrainable_variables
B	variables
imetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fregularization_losses
jnon_trainable_variables

klayers
Gtrainable_variables
H	variables
lmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
	7"
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	qtotal
	rcount
s
_fn_kwargs
t_updates
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BinaryAccuracy", "name": "acc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32", "threshold": 0.5}, "input_spec": null, "activity_regularizer": null}
?
y
thresholds
ztrue_positives
{false_positives
|_updates
}regularization_losses
~trainable_variables
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Precision", "name": "precision", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "input_spec": null, "activity_regularizer": null}
?
?
thresholds
?true_positives
?false_negatives
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Recall", "name": "recall", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "input_spec": null, "activity_regularizer": null}
?$
?
thresholds
?_layers
?true_positives
?true_negatives
?false_positives
?false_negatives
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?"
_tf_keras_layer?!{"class_name": "AUC", "name": "auc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593]}, "input_spec": null, "activity_regularizer": null}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
uregularization_losses
?non_trainable_variables
?layers
vtrainable_variables
w	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
?
}regularization_losses
?non_trainable_variables
?layers
~trainable_variables
	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
?layers
?trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
?layers
?trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
%:#
??2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
%:#
??2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95194
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95174
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95215
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95247?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_AllOneApiClassifier_layer_call_fn_95257
3__inference_AllOneApiClassifier_layer_call_fn_95225?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference__wrapped_model_94985?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *;?8
6?3
zero_padding2d_input?????????dd
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_zero_padding2d_layer_call_fn_95003?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_95017?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
&__inference_conv2d_layer_call_fn_95028?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_zero_padding2d_1_layer_call_fn_95046?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv2d_1_layer_call_fn_95071?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_95088?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_95115?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_95124?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_95138?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_95149?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_activation_layer_call_and_return_conditional_losses_95160?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_activation_layer_call_fn_95169?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B=
#__inference_signature_wrapper_95278zero_padding2d_input
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 y
*__inference_activation_layer_call_fn_95169K/?,
%?"
 ?
inputs?????????
? "???????????
(__inference_conv2d_1_layer_call_fn_95071?()I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95247l()<=;?8
1?.
(?%
inputs?????????dd
p
? "%?"
?
0?????????
? ?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95194z()<=I?F
??<
6?3
zero_padding2d_input?????????dd
p
? "%?"
?
0?????????
? ?
B__inference_flatten_layer_call_and_return_conditional_losses_95115b7?4
-?*
(?%
inputs?????????22
? "'?$
?
0???????????
? ?
'__inference_flatten_layer_call_fn_95124U7?4
-?*
(?%
inputs?????????22
? "?????????????
@__inference_dense_layer_call_and_return_conditional_losses_95138^<=1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? ?
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95215l()<=;?8
1?.
(?%
inputs?????????dd
p 
? "%?"
?
0?????????
? ?
E__inference_activation_layer_call_and_return_conditional_losses_95160X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
3__inference_AllOneApiClassifier_layer_call_fn_95257m()<=I?F
??<
6?3
zero_padding2d_input?????????dd
p
? "???????????
A__inference_conv2d_layer_call_and_return_conditional_losses_95017?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_94994?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_95037?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? z
%__inference_dense_layer_call_fn_95149Q<=1?.
'?$
"?
inputs???????????
? "???????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_95079?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_AllOneApiClassifier_layer_call_fn_95225m()<=I?F
??<
6?3
zero_padding2d_input?????????dd
p 
? "???????????
N__inference_AllOneApiClassifier_layer_call_and_return_conditional_losses_95174z()<=I?F
??<
6?3
zero_padding2d_input?????????dd
p 
? "%?"
?
0?????????
? ?
0__inference_zero_padding2d_1_layer_call_fn_95046?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95060?()I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
 __inference__wrapped_model_94985?()<=E?B
;?8
6?3
zero_padding2d_input?????????dd
? "7?4
2

activation$?!

activation??????????
&__inference_conv2d_layer_call_fn_95028?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
#__inference_signature_wrapper_95278?()<=]?Z
? 
S?P
N
zero_padding2d_input6?3
zero_padding2d_input?????????dd"7?4
2

activation$?!

activation??????????
-__inference_max_pooling2d_layer_call_fn_95088?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_zero_padding2d_layer_call_fn_95003?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????