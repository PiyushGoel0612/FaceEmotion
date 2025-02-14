��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
;
Elu
features"T
activations"T"
Ttype:
2
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-0-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
RMSprop/velocity/out_layer/biasVarHandleOp*
_output_shapes
: *0

debug_name" RMSprop/velocity/out_layer/bias/*
dtype0*
shape:*0
shared_name!RMSprop/velocity/out_layer/bias
�
3RMSprop/velocity/out_layer/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/out_layer/bias*
_output_shapes
:*
dtype0
�
!RMSprop/velocity/out_layer/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/out_layer/kernel/*
dtype0*
shape:	�*2
shared_name#!RMSprop/velocity/out_layer/kernel
�
5RMSprop/velocity/out_layer/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/out_layer/kernel*
_output_shapes
:	�*
dtype0
�
!RMSprop/velocity/batchnorm_7/betaVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/batchnorm_7/beta/*
dtype0*
shape:�*2
shared_name#!RMSprop/velocity/batchnorm_7/beta
�
5RMSprop/velocity/batchnorm_7/beta/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/batchnorm_7/beta*
_output_shapes	
:�*
dtype0
�
"RMSprop/velocity/batchnorm_7/gammaVarHandleOp*
_output_shapes
: *3

debug_name%#RMSprop/velocity/batchnorm_7/gamma/*
dtype0*
shape:�*3
shared_name$"RMSprop/velocity/batchnorm_7/gamma
�
6RMSprop/velocity/batchnorm_7/gamma/Read/ReadVariableOpReadVariableOp"RMSprop/velocity/batchnorm_7/gamma*
_output_shapes	
:�*
dtype0
�
RMSprop/velocity/dense_1/biasVarHandleOp*
_output_shapes
: *.

debug_name RMSprop/velocity/dense_1/bias/*
dtype0*
shape:�*.
shared_nameRMSprop/velocity/dense_1/bias
�
1RMSprop/velocity/dense_1/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/dense_1/bias*
_output_shapes	
:�*
dtype0
�
RMSprop/velocity/dense_1/kernelVarHandleOp*
_output_shapes
: *0

debug_name" RMSprop/velocity/dense_1/kernel/*
dtype0*
shape:���*0
shared_name!RMSprop/velocity/dense_1/kernel
�
3RMSprop/velocity/dense_1/kernel/Read/ReadVariableOpReadVariableOpRMSprop/velocity/dense_1/kernel*!
_output_shapes
:���*
dtype0
�
!RMSprop/velocity/batchnorm_4/betaVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/batchnorm_4/beta/*
dtype0*
shape:�*2
shared_name#!RMSprop/velocity/batchnorm_4/beta
�
5RMSprop/velocity/batchnorm_4/beta/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/batchnorm_4/beta*
_output_shapes	
:�*
dtype0
�
"RMSprop/velocity/batchnorm_4/gammaVarHandleOp*
_output_shapes
: *3

debug_name%#RMSprop/velocity/batchnorm_4/gamma/*
dtype0*
shape:�*3
shared_name$"RMSprop/velocity/batchnorm_4/gamma
�
6RMSprop/velocity/batchnorm_4/gamma/Read/ReadVariableOpReadVariableOp"RMSprop/velocity/batchnorm_4/gamma*
_output_shapes	
:�*
dtype0
�
RMSprop/velocity/conv2d_4/biasVarHandleOp*
_output_shapes
: */

debug_name!RMSprop/velocity/conv2d_4/bias/*
dtype0*
shape:�*/
shared_name RMSprop/velocity/conv2d_4/bias
�
2RMSprop/velocity/conv2d_4/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/conv2d_4/bias*
_output_shapes	
:�*
dtype0
�
 RMSprop/velocity/conv2d_4/kernelVarHandleOp*
_output_shapes
: *1

debug_name#!RMSprop/velocity/conv2d_4/kernel/*
dtype0*
shape:��*1
shared_name" RMSprop/velocity/conv2d_4/kernel
�
4RMSprop/velocity/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp RMSprop/velocity/conv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
RMSprop/velocity/conv2d_3/biasVarHandleOp*
_output_shapes
: */

debug_name!RMSprop/velocity/conv2d_3/bias/*
dtype0*
shape:�*/
shared_name RMSprop/velocity/conv2d_3/bias
�
2RMSprop/velocity/conv2d_3/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/conv2d_3/bias*
_output_shapes	
:�*
dtype0
�
 RMSprop/velocity/conv2d_3/kernelVarHandleOp*
_output_shapes
: *1

debug_name#!RMSprop/velocity/conv2d_3/kernel/*
dtype0*
shape:@�*1
shared_name" RMSprop/velocity/conv2d_3/kernel
�
4RMSprop/velocity/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp RMSprop/velocity/conv2d_3/kernel*'
_output_shapes
:@�*
dtype0
�
!RMSprop/velocity/batchnorm_2/betaVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/batchnorm_2/beta/*
dtype0*
shape:@*2
shared_name#!RMSprop/velocity/batchnorm_2/beta
�
5RMSprop/velocity/batchnorm_2/beta/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/batchnorm_2/beta*
_output_shapes
:@*
dtype0
�
"RMSprop/velocity/batchnorm_2/gammaVarHandleOp*
_output_shapes
: *3

debug_name%#RMSprop/velocity/batchnorm_2/gamma/*
dtype0*
shape:@*3
shared_name$"RMSprop/velocity/batchnorm_2/gamma
�
6RMSprop/velocity/batchnorm_2/gamma/Read/ReadVariableOpReadVariableOp"RMSprop/velocity/batchnorm_2/gamma*
_output_shapes
:@*
dtype0
�
RMSprop/velocity/conv2d_2/biasVarHandleOp*
_output_shapes
: */

debug_name!RMSprop/velocity/conv2d_2/bias/*
dtype0*
shape:@*/
shared_name RMSprop/velocity/conv2d_2/bias
�
2RMSprop/velocity/conv2d_2/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/conv2d_2/bias*
_output_shapes
:@*
dtype0
�
 RMSprop/velocity/conv2d_2/kernelVarHandleOp*
_output_shapes
: *1

debug_name#!RMSprop/velocity/conv2d_2/kernel/*
dtype0*
shape:@@*1
shared_name" RMSprop/velocity/conv2d_2/kernel
�
4RMSprop/velocity/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp RMSprop/velocity/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
�
!RMSprop/velocity/batchnorm_1/betaVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/batchnorm_1/beta/*
dtype0*
shape:@*2
shared_name#!RMSprop/velocity/batchnorm_1/beta
�
5RMSprop/velocity/batchnorm_1/beta/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/batchnorm_1/beta*
_output_shapes
:@*
dtype0
�
"RMSprop/velocity/batchnorm_1/gammaVarHandleOp*
_output_shapes
: *3

debug_name%#RMSprop/velocity/batchnorm_1/gamma/*
dtype0*
shape:@*3
shared_name$"RMSprop/velocity/batchnorm_1/gamma
�
6RMSprop/velocity/batchnorm_1/gamma/Read/ReadVariableOpReadVariableOp"RMSprop/velocity/batchnorm_1/gamma*
_output_shapes
:@*
dtype0
�
RMSprop/velocity/conv2d_1/biasVarHandleOp*
_output_shapes
: */

debug_name!RMSprop/velocity/conv2d_1/bias/*
dtype0*
shape:@*/
shared_name RMSprop/velocity/conv2d_1/bias
�
2RMSprop/velocity/conv2d_1/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
 RMSprop/velocity/conv2d_1/kernelVarHandleOp*
_output_shapes
: *1

debug_name#!RMSprop/velocity/conv2d_1/kernel/*
dtype0*
shape:@*1
shared_name" RMSprop/velocity/conv2d_1/kernel
�
4RMSprop/velocity/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp RMSprop/velocity/conv2d_1/kernel*&
_output_shapes
:@*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
out_layer/biasVarHandleOp*
_output_shapes
: *

debug_nameout_layer/bias/*
dtype0*
shape:*
shared_nameout_layer/bias
m
"out_layer/bias/Read/ReadVariableOpReadVariableOpout_layer/bias*
_output_shapes
:*
dtype0
�
out_layer/kernelVarHandleOp*
_output_shapes
: *!

debug_nameout_layer/kernel/*
dtype0*
shape:	�*!
shared_nameout_layer/kernel
v
$out_layer/kernel/Read/ReadVariableOpReadVariableOpout_layer/kernel*
_output_shapes
:	�*
dtype0
�
batchnorm_7/moving_varianceVarHandleOp*
_output_shapes
: *,

debug_namebatchnorm_7/moving_variance/*
dtype0*
shape:�*,
shared_namebatchnorm_7/moving_variance
�
/batchnorm_7/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_7/moving_variance*
_output_shapes	
:�*
dtype0
�
batchnorm_7/moving_meanVarHandleOp*
_output_shapes
: *(

debug_namebatchnorm_7/moving_mean/*
dtype0*
shape:�*(
shared_namebatchnorm_7/moving_mean
�
+batchnorm_7/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_7/moving_mean*
_output_shapes	
:�*
dtype0
�
batchnorm_7/betaVarHandleOp*
_output_shapes
: *!

debug_namebatchnorm_7/beta/*
dtype0*
shape:�*!
shared_namebatchnorm_7/beta
r
$batchnorm_7/beta/Read/ReadVariableOpReadVariableOpbatchnorm_7/beta*
_output_shapes	
:�*
dtype0
�
batchnorm_7/gammaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_7/gamma/*
dtype0*
shape:�*"
shared_namebatchnorm_7/gamma
t
%batchnorm_7/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_7/gamma*
_output_shapes	
:�*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:���*
shared_namedense_1/kernel
t
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_output_shapes
:���*
dtype0
�
batchnorm_4/moving_varianceVarHandleOp*
_output_shapes
: *,

debug_namebatchnorm_4/moving_variance/*
dtype0*
shape:�*,
shared_namebatchnorm_4/moving_variance
�
/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_variance*
_output_shapes	
:�*
dtype0
�
batchnorm_4/moving_meanVarHandleOp*
_output_shapes
: *(

debug_namebatchnorm_4/moving_mean/*
dtype0*
shape:�*(
shared_namebatchnorm_4/moving_mean
�
+batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_mean*
_output_shapes	
:�*
dtype0
�
batchnorm_4/betaVarHandleOp*
_output_shapes
: *!

debug_namebatchnorm_4/beta/*
dtype0*
shape:�*!
shared_namebatchnorm_4/beta
r
$batchnorm_4/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta*
_output_shapes	
:�*
dtype0
�
batchnorm_4/gammaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_4/gamma/*
dtype0*
shape:�*"
shared_namebatchnorm_4/gamma
t
%batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_4/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_4/bias/*
dtype0*
shape:�*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:�*
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_4/kernel/*
dtype0*
shape:��* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
conv2d_3/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_3/bias/*
dtype0*
shape:�*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:�*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_3/kernel/*
dtype0*
shape:@�* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@�*
dtype0
�
batchnorm_2/moving_varianceVarHandleOp*
_output_shapes
: *,

debug_namebatchnorm_2/moving_variance/*
dtype0*
shape:@*,
shared_namebatchnorm_2/moving_variance
�
/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_variance*
_output_shapes
:@*
dtype0
�
batchnorm_2/moving_meanVarHandleOp*
_output_shapes
: *(

debug_namebatchnorm_2/moving_mean/*
dtype0*
shape:@*(
shared_namebatchnorm_2/moving_mean

+batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_mean*
_output_shapes
:@*
dtype0
�
batchnorm_2/betaVarHandleOp*
_output_shapes
: *!

debug_namebatchnorm_2/beta/*
dtype0*
shape:@*!
shared_namebatchnorm_2/beta
q
$batchnorm_2/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta*
_output_shapes
:@*
dtype0
�
batchnorm_2/gammaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_2/gamma/*
dtype0*
shape:@*"
shared_namebatchnorm_2/gamma
s
%batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma*
_output_shapes
:@*
dtype0
�
conv2d_2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_2/bias/*
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_2/kernel/*
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
�
batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *,

debug_namebatchnorm_1/moving_variance/*
dtype0*
shape:@*,
shared_namebatchnorm_1/moving_variance
�
/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
:@*
dtype0
�
batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *(

debug_namebatchnorm_1/moving_mean/*
dtype0*
shape:@*(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
:@*
dtype0
�
batchnorm_1/betaVarHandleOp*
_output_shapes
: *!

debug_namebatchnorm_1/beta/*
dtype0*
shape:@*!
shared_namebatchnorm_1/beta
q
$batchnorm_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta*
_output_shapes
:@*
dtype0
�
batchnorm_1/gammaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_1/gamma/*
dtype0*
shape:@*"
shared_namebatchnorm_1/gamma
s
%batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma*
_output_shapes
:@*
dtype0
�
conv2d_1/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_1/bias/*
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_1/kernel/*
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_conv2d_1_inputPlaceholder*/
_output_shapes
:���������00*
dtype0*$
shape:���������00
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_1_inputconv2d_1/kernelconv2d_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_variancedense_1/kerneldense_1/biasbatchnorm_7/moving_variancebatchnorm_7/gammabatchnorm_7/moving_meanbatchnorm_7/betaout_layer/kernelout_layer/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_40774

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*˃
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'axis
	(gamma
)beta
*moving_mean
+moving_variance*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator* 
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator* 
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
0
1
(2
)3
*4
+5
26
37
<8
=9
>10
?11
M12
N13
V14
W15
`16
a17
b18
c19
w20
x21
�22
�23
�24
�25
�26
�27*
�
0
1
(2
)3
24
35
<6
=7
M8
N9
V10
W11
`12
a13
w14
x15
�16
�17
�18
�19*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�_velocities
�
_momentums
�_average_gradients
�_update_step_xla*

�serving_default* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
(0
)1
*2
+3*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
<0
=1
>2
?3*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

V0
W1*

V0
W1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
`0
a1
b2
c3*

`0
a1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_4/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_4/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_4/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_4/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_7/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_7/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_7/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_7/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEout_layer/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEout_layer/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
>
*0
+1
>2
?3
b4
c5
�6
�7*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
* 
* 
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

b0
c1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
ke
VARIABLE_VALUE RMSprop/velocity/conv2d_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUERMSprop/velocity/conv2d_1/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"RMSprop/velocity/batchnorm_1/gamma1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!RMSprop/velocity/batchnorm_1/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE RMSprop/velocity/conv2d_2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUERMSprop/velocity/conv2d_2/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"RMSprop/velocity/batchnorm_2/gamma1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!RMSprop/velocity/batchnorm_2/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE RMSprop/velocity/conv2d_3/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUERMSprop/velocity/conv2d_3/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE RMSprop/velocity/conv2d_4/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUERMSprop/velocity/conv2d_4/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"RMSprop/velocity/batchnorm_4/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/batchnorm_4/beta2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUERMSprop/velocity/dense_1/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUERMSprop/velocity/dense_1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"RMSprop/velocity/batchnorm_7/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/batchnorm_7/beta2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/out_layer/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUERMSprop/velocity/out_layer/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_variancedense_1/kerneldense_1/biasbatchnorm_7/gammabatchnorm_7/betabatchnorm_7/moving_meanbatchnorm_7/moving_varianceout_layer/kernelout_layer/bias	iterationlearning_rate RMSprop/velocity/conv2d_1/kernelRMSprop/velocity/conv2d_1/bias"RMSprop/velocity/batchnorm_1/gamma!RMSprop/velocity/batchnorm_1/beta RMSprop/velocity/conv2d_2/kernelRMSprop/velocity/conv2d_2/bias"RMSprop/velocity/batchnorm_2/gamma!RMSprop/velocity/batchnorm_2/beta RMSprop/velocity/conv2d_3/kernelRMSprop/velocity/conv2d_3/bias RMSprop/velocity/conv2d_4/kernelRMSprop/velocity/conv2d_4/bias"RMSprop/velocity/batchnorm_4/gamma!RMSprop/velocity/batchnorm_4/betaRMSprop/velocity/dense_1/kernelRMSprop/velocity/dense_1/bias"RMSprop/velocity/batchnorm_7/gamma!RMSprop/velocity/batchnorm_7/beta!RMSprop/velocity/out_layer/kernelRMSprop/velocity/out_layer/biastotal_1count_1totalcountConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_41698
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_variancedense_1/kerneldense_1/biasbatchnorm_7/gammabatchnorm_7/betabatchnorm_7/moving_meanbatchnorm_7/moving_varianceout_layer/kernelout_layer/bias	iterationlearning_rate RMSprop/velocity/conv2d_1/kernelRMSprop/velocity/conv2d_1/bias"RMSprop/velocity/batchnorm_1/gamma!RMSprop/velocity/batchnorm_1/beta RMSprop/velocity/conv2d_2/kernelRMSprop/velocity/conv2d_2/bias"RMSprop/velocity/batchnorm_2/gamma!RMSprop/velocity/batchnorm_2/beta RMSprop/velocity/conv2d_3/kernelRMSprop/velocity/conv2d_3/bias RMSprop/velocity/conv2d_4/kernelRMSprop/velocity/conv2d_4/bias"RMSprop/velocity/batchnorm_4/gamma!RMSprop/velocity/batchnorm_4/betaRMSprop/velocity/dense_1/kernelRMSprop/velocity/dense_1/bias"RMSprop/velocity/batchnorm_7/gamma!RMSprop/velocity/batchnorm_7/beta!RMSprop/velocity/out_layer/kernelRMSprop/velocity/out_layer/biastotal_1count_1totalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_41869��
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_41085

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:���������00�i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:���������00�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_conv2d_2_layer_call_fn_40965

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40266w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������00@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:%!

_user_specified_name40959:%!

_user_specified_name40961
�L
�
?__inference_DCNN_layer_call_and_return_conditional_losses_40499
conv2d_1_input(
conv2d_1_40413:@
conv2d_1_40415:@
batchnorm_1_40418:@
batchnorm_1_40420:@
batchnorm_1_40422:@
batchnorm_1_40424:@(
conv2d_2_40427:@@
conv2d_2_40429:@
batchnorm_2_40432:@
batchnorm_2_40434:@
batchnorm_2_40436:@
batchnorm_2_40438:@)
conv2d_3_40447:@�
conv2d_3_40449:	�*
conv2d_4_40452:��
conv2d_4_40454:	� 
batchnorm_4_40457:	� 
batchnorm_4_40459:	� 
batchnorm_4_40461:	� 
batchnorm_4_40463:	�"
dense_1_40473:���
dense_1_40475:	� 
batchnorm_7_40478:	� 
batchnorm_7_40480:	� 
batchnorm_7_40482:	� 
batchnorm_7_40484:	�"
out_layer_40493:	�
out_layer_40495:
identity��#batchnorm_1/StatefulPartitionedCall�#batchnorm_2/StatefulPartitionedCall�#batchnorm_4/StatefulPartitionedCall�#batchnorm_7/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!out_layer/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_40413conv2d_1_40415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40241�
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_40418batchnorm_1_40420batchnorm_1_40422batchnorm_1_40424*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39998�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_40427conv2d_2_40429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40266�
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_40432batchnorm_2_40434batchnorm_2_40436batchnorm_2_40438*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40060�
dropout_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40445�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_3_40447conv2d_3_40449*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40304�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_40452conv2d_4_40454*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40320�
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_40457batchnorm_4_40459batchnorm_4_40461batchnorm_4_40463*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40122�
dropout_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40470�
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_40353�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_40473dense_1_40475*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40365�
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_7_40478batchnorm_7_40480batchnorm_7_40482batchnorm_7_40484*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40202�
dropout_4/PartitionedCallPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_40491�
!out_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0out_layer_40493out_layer_40495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_40403y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:%!

_user_specified_name40413:%!

_user_specified_name40415:%!

_user_specified_name40418:%!

_user_specified_name40420:%!

_user_specified_name40422:%!

_user_specified_name40424:%!

_user_specified_name40427:%!

_user_specified_name40429:%	!

_user_specified_name40432:%
!

_user_specified_name40434:%!

_user_specified_name40436:%!

_user_specified_name40438:%!

_user_specified_name40447:%!

_user_specified_name40449:%!

_user_specified_name40452:%!

_user_specified_name40454:%!

_user_specified_name40457:%!

_user_specified_name40459:%!

_user_specified_name40461:%!

_user_specified_name40463:%!

_user_specified_name40473:%!

_user_specified_name40475:%!

_user_specified_name40478:%!

_user_specified_name40480:%!

_user_specified_name40482:%!

_user_specified_name40484:%!

_user_specified_name40493:%!

_user_specified_name40495
�
X
"__inference__update_step_xla_40829
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�	
�
+__inference_batchnorm_7_layer_call_fn_41251

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40202p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:%!

_user_specified_name41241:%!

_user_specified_name41243:%!

_user_specified_name41245:%!

_user_specified_name41247
�&
�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41285

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�&
�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40182

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40060

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40266

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������00@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:���������00@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
E
)__inference_dropout_4_layer_call_fn_41315

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_40491a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40122

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40202

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_40491

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
Q
"__inference__update_step_xla_40849
gradient
variable:���*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*"
_input_shapes
:���: *
	_noinline(:K G
!
_output_shapes
:���
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40304

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:���������00�i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:���������00�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41305

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
J
"__inference__update_step_xla_40789
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�	
�
+__inference_batchnorm_2_layer_call_fn_40989

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40042�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:%!

_user_specified_name40979:%!

_user_specified_name40981:%!

_user_specified_name40983:%!

_user_specified_name40985
�	
�
+__inference_batchnorm_4_layer_call_fn_41131

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40122�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:%!

_user_specified_name41121:%!

_user_specified_name41123:%!

_user_specified_name41125:%!

_user_specified_name41127
�	
�
+__inference_batchnorm_4_layer_call_fn_41118

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40104�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:%!

_user_specified_name41108:%!

_user_specified_name41110:%!

_user_specified_name41112:%!

_user_specified_name41114
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40894

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������00@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:���������00@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
V
"__inference__update_step_xla_40779
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_41225

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
K
"__inference__update_step_xla_40824
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_41327

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41149

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39980

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_40445

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������00@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������00@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�
K
"__inference__update_step_xla_40854
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
#__inference_signature_wrapper_40774
conv2d_1_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:���

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_39962o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:%!

_user_specified_name40716:%!

_user_specified_name40718:%!

_user_specified_name40720:%!

_user_specified_name40722:%!

_user_specified_name40724:%!

_user_specified_name40726:%!

_user_specified_name40728:%!

_user_specified_name40730:%	!

_user_specified_name40732:%
!

_user_specified_name40734:%!

_user_specified_name40736:%!

_user_specified_name40738:%!

_user_specified_name40740:%!

_user_specified_name40742:%!

_user_specified_name40744:%!

_user_specified_name40746:%!

_user_specified_name40748:%!

_user_specified_name40750:%!

_user_specified_name40752:%!

_user_specified_name40754:%!

_user_specified_name40756:%!

_user_specified_name40758:%!

_user_specified_name40760:%!

_user_specified_name40762:%!

_user_specified_name40764:%!

_user_specified_name40766:%!

_user_specified_name40768:%!

_user_specified_name40770
�
K
"__inference__update_step_xla_40844
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
D__inference_out_layer_layer_call_and_return_conditional_losses_41352

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_40365

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
J
"__inference__update_step_xla_40809
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
J
"__inference__update_step_xla_40804
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40104

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
J
"__inference__update_step_xla_40794
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
V
"__inference__update_step_xla_40799
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������00@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������00@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������00@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������00@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������00@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�	
�
+__inference_batchnorm_1_layer_call_fn_40920

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39998�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:%!

_user_specified_name40910:%!

_user_specified_name40912:%!

_user_specified_name40914:%!

_user_specified_name40916
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_41205

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_41065

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������00@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������00@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�
�
$__inference_DCNN_layer_call_fn_40621
conv2d_1_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:���

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_40499o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:%!

_user_specified_name40563:%!

_user_specified_name40565:%!

_user_specified_name40567:%!

_user_specified_name40569:%!

_user_specified_name40571:%!

_user_specified_name40573:%!

_user_specified_name40575:%!

_user_specified_name40577:%	!

_user_specified_name40579:%
!

_user_specified_name40581:%!

_user_specified_name40583:%!

_user_specified_name40585:%!

_user_specified_name40587:%!

_user_specified_name40589:%!

_user_specified_name40591:%!

_user_specified_name40593:%!

_user_specified_name40595:%!

_user_specified_name40597:%!

_user_specified_name40599:%!

_user_specified_name40601:%!

_user_specified_name40603:%!

_user_specified_name40605:%!

_user_specified_name40607:%!

_user_specified_name40609:%!

_user_specified_name40611:%!

_user_specified_name40613:%!

_user_specified_name40615:%!

_user_specified_name40617
�
�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40042

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_41189

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������00�Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������00�*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������00�T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:���������00�j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:���������00�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41167

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
J
"__inference__update_step_xla_40874
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
W
"__inference__update_step_xla_40819
gradient#
variable:@�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@�: *
	_noinline(:Q M
'
_output_shapes
:@�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
E
)__inference_dropout_1_layer_call_fn_41048

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40445h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������00@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�
�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41038

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
K
"__inference__update_step_xla_40834
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
(__inference_conv2d_4_layer_call_fn_41094

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40320x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������00�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs:%!

_user_specified_name41088:%!

_user_specified_name41090
�
�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39998

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
 __inference__wrapped_model_39962
conv2d_1_inputF
,dcnn_conv2d_1_conv2d_readvariableop_resource:@;
-dcnn_conv2d_1_biasadd_readvariableop_resource:@6
(dcnn_batchnorm_1_readvariableop_resource:@8
*dcnn_batchnorm_1_readvariableop_1_resource:@G
9dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_resource:@I
;dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource:@F
,dcnn_conv2d_2_conv2d_readvariableop_resource:@@;
-dcnn_conv2d_2_biasadd_readvariableop_resource:@6
(dcnn_batchnorm_2_readvariableop_resource:@8
*dcnn_batchnorm_2_readvariableop_1_resource:@G
9dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_resource:@I
;dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:@G
,dcnn_conv2d_3_conv2d_readvariableop_resource:@�<
-dcnn_conv2d_3_biasadd_readvariableop_resource:	�H
,dcnn_conv2d_4_conv2d_readvariableop_resource:��<
-dcnn_conv2d_4_biasadd_readvariableop_resource:	�7
(dcnn_batchnorm_4_readvariableop_resource:	�9
*dcnn_batchnorm_4_readvariableop_1_resource:	�H
9dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_resource:	�J
;dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource:	�@
+dcnn_dense_1_matmul_readvariableop_resource:���;
,dcnn_dense_1_biasadd_readvariableop_resource:	�A
2dcnn_batchnorm_7_batchnorm_readvariableop_resource:	�E
6dcnn_batchnorm_7_batchnorm_mul_readvariableop_resource:	�C
4dcnn_batchnorm_7_batchnorm_readvariableop_1_resource:	�C
4dcnn_batchnorm_7_batchnorm_readvariableop_2_resource:	�@
-dcnn_out_layer_matmul_readvariableop_resource:	�<
.dcnn_out_layer_biasadd_readvariableop_resource:
identity��0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp�2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1�DCNN/batchnorm_1/ReadVariableOp�!DCNN/batchnorm_1/ReadVariableOp_1�0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp�2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1�DCNN/batchnorm_2/ReadVariableOp�!DCNN/batchnorm_2/ReadVariableOp_1�0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp�2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1�DCNN/batchnorm_4/ReadVariableOp�!DCNN/batchnorm_4/ReadVariableOp_1�)DCNN/batchnorm_7/batchnorm/ReadVariableOp�+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1�+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2�-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp�$DCNN/conv2d_1/BiasAdd/ReadVariableOp�#DCNN/conv2d_1/Conv2D/ReadVariableOp�$DCNN/conv2d_2/BiasAdd/ReadVariableOp�#DCNN/conv2d_2/Conv2D/ReadVariableOp�$DCNN/conv2d_3/BiasAdd/ReadVariableOp�#DCNN/conv2d_3/Conv2D/ReadVariableOp�$DCNN/conv2d_4/BiasAdd/ReadVariableOp�#DCNN/conv2d_4/Conv2D/ReadVariableOp�#DCNN/dense_1/BiasAdd/ReadVariableOp�"DCNN/dense_1/MatMul/ReadVariableOp�%DCNN/out_layer/BiasAdd/ReadVariableOp�$DCNN/out_layer/MatMul/ReadVariableOp�
#DCNN/conv2d_1/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
DCNN/conv2d_1/Conv2DConv2Dconv2d_1_input+DCNN/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
�
$DCNN/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
DCNN/conv2d_1/BiasAddBiasAddDCNN/conv2d_1/Conv2D:output:0,DCNN/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@r
DCNN/conv2d_1/EluEluDCNN/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������00@�
DCNN/batchnorm_1/ReadVariableOpReadVariableOp(dcnn_batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
!DCNN/batchnorm_1/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
!DCNN/batchnorm_1/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_1/Elu:activations:0'DCNN/batchnorm_1/ReadVariableOp:value:0)DCNN/batchnorm_1/ReadVariableOp_1:value:08DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00@:@:@:@:@:*
epsilon%o�:*
is_training( �
#DCNN/conv2d_2/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
DCNN/conv2d_2/Conv2DConv2D%DCNN/batchnorm_1/FusedBatchNormV3:y:0+DCNN/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
�
$DCNN/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
DCNN/conv2d_2/BiasAddBiasAddDCNN/conv2d_2/Conv2D:output:0,DCNN/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@r
DCNN/conv2d_2/EluEluDCNN/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������00@�
DCNN/batchnorm_2/ReadVariableOpReadVariableOp(dcnn_batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
!DCNN/batchnorm_2/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
!DCNN/batchnorm_2/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_2/Elu:activations:0'DCNN/batchnorm_2/ReadVariableOp:value:0)DCNN/batchnorm_2/ReadVariableOp_1:value:08DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00@:@:@:@:@:*
epsilon%o�:*
is_training( �
DCNN/dropout_1/IdentityIdentity%DCNN/batchnorm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������00@�
#DCNN/conv2d_3/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
DCNN/conv2d_3/Conv2DConv2D DCNN/dropout_1/Identity:output:0+DCNN/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
�
$DCNN/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
DCNN/conv2d_3/BiasAddBiasAddDCNN/conv2d_3/Conv2D:output:0,DCNN/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�s
DCNN/conv2d_3/EluEluDCNN/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������00��
#DCNN/conv2d_4/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
DCNN/conv2d_4/Conv2DConv2DDCNN/conv2d_3/Elu:activations:0+DCNN/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
�
$DCNN/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
DCNN/conv2d_4/BiasAddBiasAddDCNN/conv2d_4/Conv2D:output:0,DCNN/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�s
DCNN/conv2d_4/EluEluDCNN/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:���������00��
DCNN/batchnorm_4/ReadVariableOpReadVariableOp(dcnn_batchnorm_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!DCNN/batchnorm_4/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
!DCNN/batchnorm_4/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_4/Elu:activations:0'DCNN/batchnorm_4/ReadVariableOp:value:0)DCNN/batchnorm_4/ReadVariableOp_1:value:08DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������00�:�:�:�:�:*
epsilon%o�:*
is_training( �
DCNN/dropout_2/IdentityIdentity%DCNN/batchnorm_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������00�c
DCNN/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � �
DCNN/flatten/ReshapeReshape DCNN/dropout_2/Identity:output:0DCNN/flatten/Const:output:0*
T0*)
_output_shapes
:������������
"DCNN/dense_1/MatMul/ReadVariableOpReadVariableOp+dcnn_dense_1_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
DCNN/dense_1/MatMulMatMulDCNN/flatten/Reshape:output:0*DCNN/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#DCNN/dense_1/BiasAdd/ReadVariableOpReadVariableOp,dcnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
DCNN/dense_1/BiasAddBiasAddDCNN/dense_1/MatMul:product:0+DCNN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
DCNN/dense_1/EluEluDCNN/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)DCNN/batchnorm_7/batchnorm/ReadVariableOpReadVariableOp2dcnn_batchnorm_7_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0e
 DCNN/batchnorm_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
DCNN/batchnorm_7/batchnorm/addAddV21DCNN/batchnorm_7/batchnorm/ReadVariableOp:value:0)DCNN/batchnorm_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�s
 DCNN/batchnorm_7/batchnorm/RsqrtRsqrt"DCNN/batchnorm_7/batchnorm/add:z:0*
T0*
_output_shapes	
:��
-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOpReadVariableOp6dcnn_batchnorm_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
DCNN/batchnorm_7/batchnorm/mulMul$DCNN/batchnorm_7/batchnorm/Rsqrt:y:05DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
 DCNN/batchnorm_7/batchnorm/mul_1MulDCNN/dense_1/Elu:activations:0"DCNN/batchnorm_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1ReadVariableOp4dcnn_batchnorm_7_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
 DCNN/batchnorm_7/batchnorm/mul_2Mul3DCNN/batchnorm_7/batchnorm/ReadVariableOp_1:value:0"DCNN/batchnorm_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2ReadVariableOp4dcnn_batchnorm_7_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
DCNN/batchnorm_7/batchnorm/subSub3DCNN/batchnorm_7/batchnorm/ReadVariableOp_2:value:0$DCNN/batchnorm_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
 DCNN/batchnorm_7/batchnorm/add_1AddV2$DCNN/batchnorm_7/batchnorm/mul_1:z:0"DCNN/batchnorm_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������|
DCNN/dropout_4/IdentityIdentity$DCNN/batchnorm_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
$DCNN/out_layer/MatMul/ReadVariableOpReadVariableOp-dcnn_out_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
DCNN/out_layer/MatMulMatMul DCNN/dropout_4/Identity:output:0,DCNN/out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%DCNN/out_layer/BiasAdd/ReadVariableOpReadVariableOp.dcnn_out_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
DCNN/out_layer/BiasAddBiasAddDCNN/out_layer/MatMul:product:0-DCNN/out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
DCNN/out_layer/SoftmaxSoftmaxDCNN/out_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
IdentityIdentity DCNN/out_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp1^DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_1/ReadVariableOp"^DCNN/batchnorm_1/ReadVariableOp_11^DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_2/ReadVariableOp"^DCNN/batchnorm_2/ReadVariableOp_11^DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_4/ReadVariableOp"^DCNN/batchnorm_4/ReadVariableOp_1*^DCNN/batchnorm_7/batchnorm/ReadVariableOp,^DCNN/batchnorm_7/batchnorm/ReadVariableOp_1,^DCNN/batchnorm_7/batchnorm/ReadVariableOp_2.^DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp%^DCNN/conv2d_1/BiasAdd/ReadVariableOp$^DCNN/conv2d_1/Conv2D/ReadVariableOp%^DCNN/conv2d_2/BiasAdd/ReadVariableOp$^DCNN/conv2d_2/Conv2D/ReadVariableOp%^DCNN/conv2d_3/BiasAdd/ReadVariableOp$^DCNN/conv2d_3/Conv2D/ReadVariableOp%^DCNN/conv2d_4/BiasAdd/ReadVariableOp$^DCNN/conv2d_4/Conv2D/ReadVariableOp$^DCNN/dense_1/BiasAdd/ReadVariableOp#^DCNN/dense_1/MatMul/ReadVariableOp&^DCNN/out_layer/BiasAdd/ReadVariableOp%^DCNN/out_layer/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_1/ReadVariableOpDCNN/batchnorm_1/ReadVariableOp2F
!DCNN/batchnorm_1/ReadVariableOp_1!DCNN/batchnorm_1/ReadVariableOp_12d
0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_2/ReadVariableOpDCNN/batchnorm_2/ReadVariableOp2F
!DCNN/batchnorm_2/ReadVariableOp_1!DCNN/batchnorm_2/ReadVariableOp_12d
0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_4/ReadVariableOpDCNN/batchnorm_4/ReadVariableOp2F
!DCNN/batchnorm_4/ReadVariableOp_1!DCNN/batchnorm_4/ReadVariableOp_12V
)DCNN/batchnorm_7/batchnorm/ReadVariableOp)DCNN/batchnorm_7/batchnorm/ReadVariableOp2Z
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1+DCNN/batchnorm_7/batchnorm/ReadVariableOp_12Z
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2+DCNN/batchnorm_7/batchnorm/ReadVariableOp_22^
-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp2L
$DCNN/conv2d_1/BiasAdd/ReadVariableOp$DCNN/conv2d_1/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_1/Conv2D/ReadVariableOp#DCNN/conv2d_1/Conv2D/ReadVariableOp2L
$DCNN/conv2d_2/BiasAdd/ReadVariableOp$DCNN/conv2d_2/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_2/Conv2D/ReadVariableOp#DCNN/conv2d_2/Conv2D/ReadVariableOp2L
$DCNN/conv2d_3/BiasAdd/ReadVariableOp$DCNN/conv2d_3/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_3/Conv2D/ReadVariableOp#DCNN/conv2d_3/Conv2D/ReadVariableOp2L
$DCNN/conv2d_4/BiasAdd/ReadVariableOp$DCNN/conv2d_4/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_4/Conv2D/ReadVariableOp#DCNN/conv2d_4/Conv2D/ReadVariableOp2J
#DCNN/dense_1/BiasAdd/ReadVariableOp#DCNN/dense_1/BiasAdd/ReadVariableOp2H
"DCNN/dense_1/MatMul/ReadVariableOp"DCNN/dense_1/MatMul/ReadVariableOp2N
%DCNN/out_layer/BiasAdd/ReadVariableOp%DCNN/out_layer/BiasAdd/ReadVariableOp2L
$DCNN/out_layer/MatMul/ReadVariableOp$DCNN/out_layer/MatMul/ReadVariableOp:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
K
"__inference__update_step_xla_40839
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
J
"__inference__update_step_xla_40784
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
b
)__inference_dropout_1_layer_call_fn_41043

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40292w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������00@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�
K
"__inference__update_step_xla_40859
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41020

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
b
)__inference_dropout_4_layer_call_fn_41310

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_40391p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_40883

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40241w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������00@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������00
 
_user_specified_nameinputs:%!

_user_specified_name40877:%!

_user_specified_name40879
�P
�
?__inference_DCNN_layer_call_and_return_conditional_losses_40410
conv2d_1_input(
conv2d_1_40242:@
conv2d_1_40244:@
batchnorm_1_40247:@
batchnorm_1_40249:@
batchnorm_1_40251:@
batchnorm_1_40253:@(
conv2d_2_40267:@@
conv2d_2_40269:@
batchnorm_2_40272:@
batchnorm_2_40274:@
batchnorm_2_40276:@
batchnorm_2_40278:@)
conv2d_3_40305:@�
conv2d_3_40307:	�*
conv2d_4_40321:��
conv2d_4_40323:	� 
batchnorm_4_40326:	� 
batchnorm_4_40328:	� 
batchnorm_4_40330:	� 
batchnorm_4_40332:	�"
dense_1_40366:���
dense_1_40368:	� 
batchnorm_7_40371:	� 
batchnorm_7_40373:	� 
batchnorm_7_40375:	� 
batchnorm_7_40377:	�"
out_layer_40404:	�
out_layer_40406:
identity��#batchnorm_1/StatefulPartitionedCall�#batchnorm_2/StatefulPartitionedCall�#batchnorm_4/StatefulPartitionedCall�#batchnorm_7/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!out_layer/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_40242conv2d_1_40244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40241�
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_40247batchnorm_1_40249batchnorm_1_40251batchnorm_1_40253*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39980�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_40267conv2d_2_40269*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40266�
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_40272batchnorm_2_40274batchnorm_2_40276batchnorm_2_40278*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40042�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40292�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_3_40305conv2d_3_40307*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40304�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_40321conv2d_4_40323*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40320�
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_40326batchnorm_4_40328batchnorm_4_40330batchnorm_4_40332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_40104�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40346�
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_40353�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_40366dense_1_40368*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40365�
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_7_40371batchnorm_7_40373batchnorm_7_40375batchnorm_7_40377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40182�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_40391�
!out_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0out_layer_40404out_layer_40406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_40403y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:%!

_user_specified_name40242:%!

_user_specified_name40244:%!

_user_specified_name40247:%!

_user_specified_name40249:%!

_user_specified_name40251:%!

_user_specified_name40253:%!

_user_specified_name40267:%!

_user_specified_name40269:%	!

_user_specified_name40272:%
!

_user_specified_name40274:%!

_user_specified_name40276:%!

_user_specified_name40278:%!

_user_specified_name40305:%!

_user_specified_name40307:%!

_user_specified_name40321:%!

_user_specified_name40323:%!

_user_specified_name40326:%!

_user_specified_name40328:%!

_user_specified_name40330:%!

_user_specified_name40332:%!

_user_specified_name40366:%!

_user_specified_name40368:%!

_user_specified_name40371:%!

_user_specified_name40373:%!

_user_specified_name40375:%!

_user_specified_name40377:%!

_user_specified_name40404:%!

_user_specified_name40406
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40976

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������00@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:���������00@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_41332

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_41194

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:���������00�d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������00�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
��
�#
!__inference__traced_restore_41869
file_prefix:
 assignvariableop_conv2d_1_kernel:@.
 assignvariableop_1_conv2d_1_bias:@2
$assignvariableop_2_batchnorm_1_gamma:@1
#assignvariableop_3_batchnorm_1_beta:@8
*assignvariableop_4_batchnorm_1_moving_mean:@<
.assignvariableop_5_batchnorm_1_moving_variance:@<
"assignvariableop_6_conv2d_2_kernel:@@.
 assignvariableop_7_conv2d_2_bias:@2
$assignvariableop_8_batchnorm_2_gamma:@1
#assignvariableop_9_batchnorm_2_beta:@9
+assignvariableop_10_batchnorm_2_moving_mean:@=
/assignvariableop_11_batchnorm_2_moving_variance:@>
#assignvariableop_12_conv2d_3_kernel:@�0
!assignvariableop_13_conv2d_3_bias:	�?
#assignvariableop_14_conv2d_4_kernel:��0
!assignvariableop_15_conv2d_4_bias:	�4
%assignvariableop_16_batchnorm_4_gamma:	�3
$assignvariableop_17_batchnorm_4_beta:	�:
+assignvariableop_18_batchnorm_4_moving_mean:	�>
/assignvariableop_19_batchnorm_4_moving_variance:	�7
"assignvariableop_20_dense_1_kernel:���/
 assignvariableop_21_dense_1_bias:	�4
%assignvariableop_22_batchnorm_7_gamma:	�3
$assignvariableop_23_batchnorm_7_beta:	�:
+assignvariableop_24_batchnorm_7_moving_mean:	�>
/assignvariableop_25_batchnorm_7_moving_variance:	�7
$assignvariableop_26_out_layer_kernel:	�0
"assignvariableop_27_out_layer_bias:'
assignvariableop_28_iteration:	 +
!assignvariableop_29_learning_rate: N
4assignvariableop_30_rmsprop_velocity_conv2d_1_kernel:@@
2assignvariableop_31_rmsprop_velocity_conv2d_1_bias:@D
6assignvariableop_32_rmsprop_velocity_batchnorm_1_gamma:@C
5assignvariableop_33_rmsprop_velocity_batchnorm_1_beta:@N
4assignvariableop_34_rmsprop_velocity_conv2d_2_kernel:@@@
2assignvariableop_35_rmsprop_velocity_conv2d_2_bias:@D
6assignvariableop_36_rmsprop_velocity_batchnorm_2_gamma:@C
5assignvariableop_37_rmsprop_velocity_batchnorm_2_beta:@O
4assignvariableop_38_rmsprop_velocity_conv2d_3_kernel:@�A
2assignvariableop_39_rmsprop_velocity_conv2d_3_bias:	�P
4assignvariableop_40_rmsprop_velocity_conv2d_4_kernel:��A
2assignvariableop_41_rmsprop_velocity_conv2d_4_bias:	�E
6assignvariableop_42_rmsprop_velocity_batchnorm_4_gamma:	�D
5assignvariableop_43_rmsprop_velocity_batchnorm_4_beta:	�H
3assignvariableop_44_rmsprop_velocity_dense_1_kernel:���@
1assignvariableop_45_rmsprop_velocity_dense_1_bias:	�E
6assignvariableop_46_rmsprop_velocity_batchnorm_7_gamma:	�D
5assignvariableop_47_rmsprop_velocity_batchnorm_7_beta:	�H
5assignvariableop_48_rmsprop_velocity_out_layer_kernel:	�A
3assignvariableop_49_rmsprop_velocity_out_layer_bias:%
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_4_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_batchnorm_4_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_batchnorm_4_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_batchnorm_4_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batchnorm_4_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_batchnorm_7_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_batchnorm_7_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_batchnorm_7_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batchnorm_7_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_out_layer_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_out_layer_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_iterationIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_rmsprop_velocity_conv2d_1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp2assignvariableop_31_rmsprop_velocity_conv2d_1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp6assignvariableop_32_rmsprop_velocity_batchnorm_1_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_rmsprop_velocity_batchnorm_1_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp4assignvariableop_34_rmsprop_velocity_conv2d_2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp2assignvariableop_35_rmsprop_velocity_conv2d_2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp6assignvariableop_36_rmsprop_velocity_batchnorm_2_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_rmsprop_velocity_batchnorm_2_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp4assignvariableop_38_rmsprop_velocity_conv2d_3_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp2assignvariableop_39_rmsprop_velocity_conv2d_3_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp4assignvariableop_40_rmsprop_velocity_conv2d_4_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp2assignvariableop_41_rmsprop_velocity_conv2d_4_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_rmsprop_velocity_batchnorm_4_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp5assignvariableop_43_rmsprop_velocity_batchnorm_4_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp3assignvariableop_44_rmsprop_velocity_dense_1_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp1assignvariableop_45_rmsprop_velocity_dense_1_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_rmsprop_velocity_batchnorm_7_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp5assignvariableop_47_rmsprop_velocity_batchnorm_7_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_rmsprop_velocity_out_layer_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp3assignvariableop_49_rmsprop_velocity_out_layer_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:1-
+
_user_specified_namebatchnorm_1/gamma:0,
*
_user_specified_namebatchnorm_1/beta:73
1
_user_specified_namebatchnorm_1/moving_mean:;7
5
_user_specified_namebatchnorm_1/moving_variance:/+
)
_user_specified_nameconv2d_2/kernel:-)
'
_user_specified_nameconv2d_2/bias:1	-
+
_user_specified_namebatchnorm_2/gamma:0
,
*
_user_specified_namebatchnorm_2/beta:73
1
_user_specified_namebatchnorm_2/moving_mean:;7
5
_user_specified_namebatchnorm_2/moving_variance:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:1-
+
_user_specified_namebatchnorm_4/gamma:0,
*
_user_specified_namebatchnorm_4/beta:73
1
_user_specified_namebatchnorm_4/moving_mean:;7
5
_user_specified_namebatchnorm_4/moving_variance:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:1-
+
_user_specified_namebatchnorm_7/gamma:0,
*
_user_specified_namebatchnorm_7/beta:73
1
_user_specified_namebatchnorm_7/moving_mean:;7
5
_user_specified_namebatchnorm_7/moving_variance:0,
*
_user_specified_nameout_layer/kernel:.*
(
_user_specified_nameout_layer/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:@<
:
_user_specified_name" RMSprop/velocity/conv2d_1/kernel:> :
8
_user_specified_name RMSprop/velocity/conv2d_1/bias:B!>
<
_user_specified_name$"RMSprop/velocity/batchnorm_1/gamma:A"=
;
_user_specified_name#!RMSprop/velocity/batchnorm_1/beta:@#<
:
_user_specified_name" RMSprop/velocity/conv2d_2/kernel:>$:
8
_user_specified_name RMSprop/velocity/conv2d_2/bias:B%>
<
_user_specified_name$"RMSprop/velocity/batchnorm_2/gamma:A&=
;
_user_specified_name#!RMSprop/velocity/batchnorm_2/beta:@'<
:
_user_specified_name" RMSprop/velocity/conv2d_3/kernel:>(:
8
_user_specified_name RMSprop/velocity/conv2d_3/bias:@)<
:
_user_specified_name" RMSprop/velocity/conv2d_4/kernel:>*:
8
_user_specified_name RMSprop/velocity/conv2d_4/bias:B+>
<
_user_specified_name$"RMSprop/velocity/batchnorm_4/gamma:A,=
;
_user_specified_name#!RMSprop/velocity/batchnorm_4/beta:?-;
9
_user_specified_name!RMSprop/velocity/dense_1/kernel:=.9
7
_user_specified_nameRMSprop/velocity/dense_1/bias:B/>
<
_user_specified_name$"RMSprop/velocity/batchnorm_7/gamma:A0=
;
_user_specified_name#!RMSprop/velocity/batchnorm_7/beta:A1=
;
_user_specified_name#!RMSprop/velocity/out_layer/kernel:?2;
9
_user_specified_name!RMSprop/velocity/out_layer/bias:'3#
!
_user_specified_name	total_1:'4#
!
_user_specified_name	count_1:%5!

_user_specified_nametotal:%6!

_user_specified_namecount
�
�
)__inference_out_layer_layer_call_fn_41341

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_40403o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:%!

_user_specified_name41335:%!

_user_specified_name41337
�
�
$__inference_DCNN_layer_call_fn_40560
conv2d_1_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:���

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_40410o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������00: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������00
(
_user_specified_nameconv2d_1_input:%!

_user_specified_name40502:%!

_user_specified_name40504:%!

_user_specified_name40506:%!

_user_specified_name40508:%!

_user_specified_name40510:%!

_user_specified_name40512:%!

_user_specified_name40514:%!

_user_specified_name40516:%	!

_user_specified_name40518:%
!

_user_specified_name40520:%!

_user_specified_name40522:%!

_user_specified_name40524:%!

_user_specified_name40526:%!

_user_specified_name40528:%!

_user_specified_name40530:%!

_user_specified_name40532:%!

_user_specified_name40534:%!

_user_specified_name40536:%!

_user_specified_name40538:%!

_user_specified_name40540:%!

_user_specified_name40542:%!

_user_specified_name40544:%!

_user_specified_name40546:%!

_user_specified_name40548:%!

_user_specified_name40550:%!

_user_specified_name40552:%!

_user_specified_name40554:%!

_user_specified_name40556
�
J
"__inference__update_step_xla_40814
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
C
'__inference_flatten_layer_call_fn_41199

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_40353b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�	
�
+__inference_batchnorm_7_layer_call_fn_41238

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_40182p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:%!

_user_specified_name41228:%!

_user_specified_name41230:%!

_user_specified_name41232:%!

_user_specified_name41234
�
E
)__inference_dropout_2_layer_call_fn_41177

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40470i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������00�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40956

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ƞ
�2
__inference__traced_save_41698
file_prefix@
&read_disablecopyonread_conv2d_1_kernel:@4
&read_1_disablecopyonread_conv2d_1_bias:@8
*read_2_disablecopyonread_batchnorm_1_gamma:@7
)read_3_disablecopyonread_batchnorm_1_beta:@>
0read_4_disablecopyonread_batchnorm_1_moving_mean:@B
4read_5_disablecopyonread_batchnorm_1_moving_variance:@B
(read_6_disablecopyonread_conv2d_2_kernel:@@4
&read_7_disablecopyonread_conv2d_2_bias:@8
*read_8_disablecopyonread_batchnorm_2_gamma:@7
)read_9_disablecopyonread_batchnorm_2_beta:@?
1read_10_disablecopyonread_batchnorm_2_moving_mean:@C
5read_11_disablecopyonread_batchnorm_2_moving_variance:@D
)read_12_disablecopyonread_conv2d_3_kernel:@�6
'read_13_disablecopyonread_conv2d_3_bias:	�E
)read_14_disablecopyonread_conv2d_4_kernel:��6
'read_15_disablecopyonread_conv2d_4_bias:	�:
+read_16_disablecopyonread_batchnorm_4_gamma:	�9
*read_17_disablecopyonread_batchnorm_4_beta:	�@
1read_18_disablecopyonread_batchnorm_4_moving_mean:	�D
5read_19_disablecopyonread_batchnorm_4_moving_variance:	�=
(read_20_disablecopyonread_dense_1_kernel:���5
&read_21_disablecopyonread_dense_1_bias:	�:
+read_22_disablecopyonread_batchnorm_7_gamma:	�9
*read_23_disablecopyonread_batchnorm_7_beta:	�@
1read_24_disablecopyonread_batchnorm_7_moving_mean:	�D
5read_25_disablecopyonread_batchnorm_7_moving_variance:	�=
*read_26_disablecopyonread_out_layer_kernel:	�6
(read_27_disablecopyonread_out_layer_bias:-
#read_28_disablecopyonread_iteration:	 1
'read_29_disablecopyonread_learning_rate: T
:read_30_disablecopyonread_rmsprop_velocity_conv2d_1_kernel:@F
8read_31_disablecopyonread_rmsprop_velocity_conv2d_1_bias:@J
<read_32_disablecopyonread_rmsprop_velocity_batchnorm_1_gamma:@I
;read_33_disablecopyonread_rmsprop_velocity_batchnorm_1_beta:@T
:read_34_disablecopyonread_rmsprop_velocity_conv2d_2_kernel:@@F
8read_35_disablecopyonread_rmsprop_velocity_conv2d_2_bias:@J
<read_36_disablecopyonread_rmsprop_velocity_batchnorm_2_gamma:@I
;read_37_disablecopyonread_rmsprop_velocity_batchnorm_2_beta:@U
:read_38_disablecopyonread_rmsprop_velocity_conv2d_3_kernel:@�G
8read_39_disablecopyonread_rmsprop_velocity_conv2d_3_bias:	�V
:read_40_disablecopyonread_rmsprop_velocity_conv2d_4_kernel:��G
8read_41_disablecopyonread_rmsprop_velocity_conv2d_4_bias:	�K
<read_42_disablecopyonread_rmsprop_velocity_batchnorm_4_gamma:	�J
;read_43_disablecopyonread_rmsprop_velocity_batchnorm_4_beta:	�N
9read_44_disablecopyonread_rmsprop_velocity_dense_1_kernel:���F
7read_45_disablecopyonread_rmsprop_velocity_dense_1_bias:	�K
<read_46_disablecopyonread_rmsprop_velocity_batchnorm_7_gamma:	�J
;read_47_disablecopyonread_rmsprop_velocity_batchnorm_7_beta:	�N
;read_48_disablecopyonread_rmsprop_velocity_out_layer_kernel:	�G
9read_49_disablecopyonread_rmsprop_velocity_out_layer_bias:+
!read_50_disablecopyonread_total_1: +
!read_51_disablecopyonread_count_1: )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: 
savev2_const
identity_109��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_batchnorm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_batchnorm_1_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_batchnorm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_batchnorm_1_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnRead0read_4_disablecopyonread_batchnorm_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp0read_4_disablecopyonread_batchnorm_1_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_batchnorm_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_batchnorm_1_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_8/DisableCopyOnReadDisableCopyOnRead*read_8_disablecopyonread_batchnorm_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp*read_8_disablecopyonread_batchnorm_2_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_batchnorm_2_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_batchnorm_2_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_10/DisableCopyOnReadDisableCopyOnRead1read_10_disablecopyonread_batchnorm_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp1read_10_disablecopyonread_batchnorm_2_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead5read_11_disablecopyonread_batchnorm_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp5read_11_disablecopyonread_batchnorm_2_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv2d_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv2d_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_conv2d_4_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_conv2d_4_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead+read_16_disablecopyonread_batchnorm_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp+read_16_disablecopyonread_batchnorm_4_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_batchnorm_4_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_batchnorm_4_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_batchnorm_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_batchnorm_4_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead5read_19_disablecopyonread_batchnorm_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp5read_19_disablecopyonread_batchnorm_4_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_dense_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0r
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���h
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*!
_output_shapes
:���{
Read_21/DisableCopyOnReadDisableCopyOnRead&read_21_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp&read_21_disablecopyonread_dense_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead+read_22_disablecopyonread_batchnorm_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp+read_22_disablecopyonread_batchnorm_7_gamma^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_batchnorm_7_beta"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_batchnorm_7_beta^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead1read_24_disablecopyonread_batchnorm_7_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp1read_24_disablecopyonread_batchnorm_7_moving_mean^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead5read_25_disablecopyonread_batchnorm_7_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp5read_25_disablecopyonread_batchnorm_7_moving_variance^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_26/DisableCopyOnReadDisableCopyOnRead*read_26_disablecopyonread_out_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp*read_26_disablecopyonread_out_layer_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	�}
Read_27/DisableCopyOnReadDisableCopyOnRead(read_27_disablecopyonread_out_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp(read_27_disablecopyonread_out_layer_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_28/DisableCopyOnReadDisableCopyOnRead#read_28_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp#read_28_disablecopyonread_iteration^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_learning_rate^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnRead:read_30_disablecopyonread_rmsprop_velocity_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp:read_30_disablecopyonread_rmsprop_velocity_conv2d_1_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_31/DisableCopyOnReadDisableCopyOnRead8read_31_disablecopyonread_rmsprop_velocity_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp8read_31_disablecopyonread_rmsprop_velocity_conv2d_1_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_32/DisableCopyOnReadDisableCopyOnRead<read_32_disablecopyonread_rmsprop_velocity_batchnorm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp<read_32_disablecopyonread_rmsprop_velocity_batchnorm_1_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_33/DisableCopyOnReadDisableCopyOnRead;read_33_disablecopyonread_rmsprop_velocity_batchnorm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp;read_33_disablecopyonread_rmsprop_velocity_batchnorm_1_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnRead:read_34_disablecopyonread_rmsprop_velocity_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp:read_34_disablecopyonread_rmsprop_velocity_conv2d_2_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_35/DisableCopyOnReadDisableCopyOnRead8read_35_disablecopyonread_rmsprop_velocity_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp8read_35_disablecopyonread_rmsprop_velocity_conv2d_2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_36/DisableCopyOnReadDisableCopyOnRead<read_36_disablecopyonread_rmsprop_velocity_batchnorm_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp<read_36_disablecopyonread_rmsprop_velocity_batchnorm_2_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead;read_37_disablecopyonread_rmsprop_velocity_batchnorm_2_beta"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp;read_37_disablecopyonread_rmsprop_velocity_batchnorm_2_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead:read_38_disablecopyonread_rmsprop_velocity_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp:read_38_disablecopyonread_rmsprop_velocity_conv2d_3_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_39/DisableCopyOnReadDisableCopyOnRead8read_39_disablecopyonread_rmsprop_velocity_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp8read_39_disablecopyonread_rmsprop_velocity_conv2d_3_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead:read_40_disablecopyonread_rmsprop_velocity_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp:read_40_disablecopyonread_rmsprop_velocity_conv2d_4_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_41/DisableCopyOnReadDisableCopyOnRead8read_41_disablecopyonread_rmsprop_velocity_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp8read_41_disablecopyonread_rmsprop_velocity_conv2d_4_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead<read_42_disablecopyonread_rmsprop_velocity_batchnorm_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp<read_42_disablecopyonread_rmsprop_velocity_batchnorm_4_gamma^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead;read_43_disablecopyonread_rmsprop_velocity_batchnorm_4_beta"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp;read_43_disablecopyonread_rmsprop_velocity_batchnorm_4_beta^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnRead9read_44_disablecopyonread_rmsprop_velocity_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp9read_44_disablecopyonread_rmsprop_velocity_dense_1_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0r
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���h
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_45/DisableCopyOnReadDisableCopyOnRead7read_45_disablecopyonread_rmsprop_velocity_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp7read_45_disablecopyonread_rmsprop_velocity_dense_1_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead<read_46_disablecopyonread_rmsprop_velocity_batchnorm_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp<read_46_disablecopyonread_rmsprop_velocity_batchnorm_7_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead;read_47_disablecopyonread_rmsprop_velocity_batchnorm_7_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp;read_47_disablecopyonread_rmsprop_velocity_batchnorm_7_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead;read_48_disablecopyonread_rmsprop_velocity_out_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp;read_48_disablecopyonread_rmsprop_velocity_out_layer_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_49/DisableCopyOnReadDisableCopyOnRead9read_49_disablecopyonread_rmsprop_velocity_out_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp9read_49_disablecopyonread_rmsprop_velocity_out_layer_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_1^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_109Identity_109:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:1-
+
_user_specified_namebatchnorm_1/gamma:0,
*
_user_specified_namebatchnorm_1/beta:73
1
_user_specified_namebatchnorm_1/moving_mean:;7
5
_user_specified_namebatchnorm_1/moving_variance:/+
)
_user_specified_nameconv2d_2/kernel:-)
'
_user_specified_nameconv2d_2/bias:1	-
+
_user_specified_namebatchnorm_2/gamma:0
,
*
_user_specified_namebatchnorm_2/beta:73
1
_user_specified_namebatchnorm_2/moving_mean:;7
5
_user_specified_namebatchnorm_2/moving_variance:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:1-
+
_user_specified_namebatchnorm_4/gamma:0,
*
_user_specified_namebatchnorm_4/beta:73
1
_user_specified_namebatchnorm_4/moving_mean:;7
5
_user_specified_namebatchnorm_4/moving_variance:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:1-
+
_user_specified_namebatchnorm_7/gamma:0,
*
_user_specified_namebatchnorm_7/beta:73
1
_user_specified_namebatchnorm_7/moving_mean:;7
5
_user_specified_namebatchnorm_7/moving_variance:0,
*
_user_specified_nameout_layer/kernel:.*
(
_user_specified_nameout_layer/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:@<
:
_user_specified_name" RMSprop/velocity/conv2d_1/kernel:> :
8
_user_specified_name RMSprop/velocity/conv2d_1/bias:B!>
<
_user_specified_name$"RMSprop/velocity/batchnorm_1/gamma:A"=
;
_user_specified_name#!RMSprop/velocity/batchnorm_1/beta:@#<
:
_user_specified_name" RMSprop/velocity/conv2d_2/kernel:>$:
8
_user_specified_name RMSprop/velocity/conv2d_2/bias:B%>
<
_user_specified_name$"RMSprop/velocity/batchnorm_2/gamma:A&=
;
_user_specified_name#!RMSprop/velocity/batchnorm_2/beta:@'<
:
_user_specified_name" RMSprop/velocity/conv2d_3/kernel:>(:
8
_user_specified_name RMSprop/velocity/conv2d_3/bias:@)<
:
_user_specified_name" RMSprop/velocity/conv2d_4/kernel:>*:
8
_user_specified_name RMSprop/velocity/conv2d_4/bias:B+>
<
_user_specified_name$"RMSprop/velocity/batchnorm_4/gamma:A,=
;
_user_specified_name#!RMSprop/velocity/batchnorm_4/beta:?-;
9
_user_specified_name!RMSprop/velocity/dense_1/kernel:=.9
7
_user_specified_nameRMSprop/velocity/dense_1/bias:B/>
<
_user_specified_name$"RMSprop/velocity/batchnorm_7/gamma:A0=
;
_user_specified_name#!RMSprop/velocity/batchnorm_7/beta:A1=
;
_user_specified_name#!RMSprop/velocity/out_layer/kernel:?2;
9
_user_specified_name!RMSprop/velocity/out_layer/bias:'3#
!
_user_specified_name	total_1:'4#
!
_user_specified_name	count_1:%5!

_user_specified_nametotal:%6!

_user_specified_namecount:=79

_output_shapes
: 

_user_specified_nameConst
�	
�
+__inference_batchnorm_1_layer_call_fn_40907

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_39980�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:%!

_user_specified_name40897:%!

_user_specified_name40899:%!

_user_specified_name40901:%!

_user_specified_name40903
�

�
D__inference_out_layer_layer_call_and_return_conditional_losses_40403

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
+__inference_batchnorm_2_layer_call_fn_41002

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_40060�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:%!

_user_specified_name40992:%!

_user_specified_name40994:%!

_user_specified_name40996:%!

_user_specified_name40998
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40241

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������00@h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:���������00@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������00
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
'__inference_dense_1_layer_call_fn_41214

inputs
unknown:���
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40365p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:%!

_user_specified_name41208:%!

_user_specified_name41210
�
K
"__inference__update_step_xla_40864
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40938

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_41105

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:���������00�i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:���������00�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_conv2d_3_layer_call_fn_41074

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40304x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������00�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������00@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs:%!

_user_specified_name41068:%!

_user_specified_name41070
�

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_40346

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������00�Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������00�*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������00�T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:���������00�j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:���������00�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_40391

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
O
"__inference__update_step_xla_40869
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:I E

_output_shapes
:	�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_40353

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_40470

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:���������00�d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������00�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_40292

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������00@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������00@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������00@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������00@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������00@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������00@:W S
/
_output_shapes
:���������00@
 
_user_specified_nameinputs
�
b
)__inference_dropout_2_layer_call_fn_41172

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40346x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������00�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������00�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40320

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������00�W
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:���������00�i
IdentityIdentityElu:activations:0^NoOp*
T0*0
_output_shapes
:���������00�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Q
conv2d_1_input?
 serving_default_conv2d_1_input:0���������00=
	out_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'axis
	(gamma
)beta
*moving_mean
+moving_variance"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
0
1
(2
)3
*4
+5
26
37
<8
=9
>10
?11
M12
N13
V14
W15
`16
a17
b18
c19
w20
x21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
0
1
(2
)3
24
35
<6
=7
M8
N9
V10
W11
`12
a13
w14
x15
�16
�17
�18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_DCNN_layer_call_fn_40560
$__inference_DCNN_layer_call_fn_40621�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_DCNN_layer_call_and_return_conditional_losses_40410
?__inference_DCNN_layer_call_and_return_conditional_losses_40499�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
 __inference__wrapped_model_39962conv2d_1_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�_velocities
�
_momentums
�_average_gradients
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_1_layer_call_fn_40883�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40894�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'@2conv2d_1/kernel
:@2conv2d_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_batchnorm_1_layer_call_fn_40907
+__inference_batchnorm_1_layer_call_fn_40920�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40938
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40956�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:@2batchnorm_1/gamma
:@2batchnorm_1/beta
':%@ (2batchnorm_1/moving_mean
+:)@ (2batchnorm_1/moving_variance
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_2_layer_call_fn_40965�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40976�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_batchnorm_2_layer_call_fn_40989
+__inference_batchnorm_2_layer_call_fn_41002�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41020
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41038�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:@2batchnorm_2/gamma
:@2batchnorm_2/beta
':%@ (2batchnorm_2/moving_mean
+:)@ (2batchnorm_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_1_layer_call_fn_41043
)__inference_dropout_1_layer_call_fn_41048�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060
D__inference_dropout_1_layer_call_and_return_conditional_losses_41065�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_3_layer_call_fn_41074�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_41085�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@�2conv2d_3/kernel
:�2conv2d_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_4_layer_call_fn_41094�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_41105�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_4/kernel
:�2conv2d_4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_batchnorm_4_layer_call_fn_41118
+__inference_batchnorm_4_layer_call_fn_41131�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41149
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41167�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 :�2batchnorm_4/gamma
:�2batchnorm_4/beta
(:&� (2batchnorm_4/moving_mean
,:*� (2batchnorm_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_2_layer_call_fn_41172
)__inference_dropout_2_layer_call_fn_41177�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_2_layer_call_and_return_conditional_losses_41189
D__inference_dropout_2_layer_call_and_return_conditional_losses_41194�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_41199�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_41205�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_41214�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_41225�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!���2dense_1/kernel
:�2dense_1/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_batchnorm_7_layer_call_fn_41238
+__inference_batchnorm_7_layer_call_fn_41251�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41285
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41305�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 :�2batchnorm_7/gamma
:�2batchnorm_7/beta
(:&� (2batchnorm_7/moving_mean
,:*� (2batchnorm_7/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_4_layer_call_fn_41310
)__inference_dropout_4_layer_call_fn_41315�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_4_layer_call_and_return_conditional_losses_41327
D__inference_dropout_4_layer_call_and_return_conditional_losses_41332�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_out_layer_layer_call_fn_41341�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_out_layer_layer_call_and_return_conditional_losses_41352�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�2out_layer/kernel
:2out_layer/bias
Z
*0
+1
>2
?3
b4
c5
�6
�7"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_DCNN_layer_call_fn_40560conv2d_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_DCNN_layer_call_fn_40621conv2d_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_DCNN_layer_call_and_return_conditional_losses_40410conv2d_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_DCNN_layer_call_and_return_conditional_losses_40499conv2d_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_192�
"__inference__update_step_xla_40779
"__inference__update_step_xla_40784
"__inference__update_step_xla_40789
"__inference__update_step_xla_40794
"__inference__update_step_xla_40799
"__inference__update_step_xla_40804
"__inference__update_step_xla_40809
"__inference__update_step_xla_40814
"__inference__update_step_xla_40819
"__inference__update_step_xla_40824
"__inference__update_step_xla_40829
"__inference__update_step_xla_40834
"__inference__update_step_xla_40839
"__inference__update_step_xla_40844
"__inference__update_step_xla_40849
"__inference__update_step_xla_40854
"__inference__update_step_xla_40859
"__inference__update_step_xla_40864
"__inference__update_step_xla_40869
"__inference__update_step_xla_40874�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15z�trace_16z�trace_17z�trace_18z�trace_19
�B�
#__inference_signature_wrapper_40774conv2d_1_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jconv2d_1_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_1_layer_call_fn_40883inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40894inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_batchnorm_1_layer_call_fn_40907inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_batchnorm_1_layer_call_fn_40920inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40938inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40956inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_2_layer_call_fn_40965inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40976inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_batchnorm_2_layer_call_fn_40989inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_batchnorm_2_layer_call_fn_41002inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41020inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41038inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_1_layer_call_fn_41043inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_1_layer_call_fn_41048inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_41065inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_3_layer_call_fn_41074inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_41085inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_4_layer_call_fn_41094inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_41105inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_batchnorm_4_layer_call_fn_41118inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_batchnorm_4_layer_call_fn_41131inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41149inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41167inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_2_layer_call_fn_41172inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_2_layer_call_fn_41177inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_2_layer_call_and_return_conditional_losses_41189inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_2_layer_call_and_return_conditional_losses_41194inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_41199inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_41205inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_41214inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_41225inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_batchnorm_7_layer_call_fn_41238inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_batchnorm_7_layer_call_fn_41251inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41285inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41305inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_4_layer_call_fn_41310inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_4_layer_call_fn_41315inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_4_layer_call_and_return_conditional_losses_41327inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_4_layer_call_and_return_conditional_losses_41332inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_out_layer_layer_call_fn_41341inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_out_layer_layer_call_and_return_conditional_losses_41352inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
8:6@2 RMSprop/velocity/conv2d_1/kernel
*:(@2RMSprop/velocity/conv2d_1/bias
.:,@2"RMSprop/velocity/batchnorm_1/gamma
-:+@2!RMSprop/velocity/batchnorm_1/beta
8:6@@2 RMSprop/velocity/conv2d_2/kernel
*:(@2RMSprop/velocity/conv2d_2/bias
.:,@2"RMSprop/velocity/batchnorm_2/gamma
-:+@2!RMSprop/velocity/batchnorm_2/beta
9:7@�2 RMSprop/velocity/conv2d_3/kernel
+:)�2RMSprop/velocity/conv2d_3/bias
::8��2 RMSprop/velocity/conv2d_4/kernel
+:)�2RMSprop/velocity/conv2d_4/bias
/:-�2"RMSprop/velocity/batchnorm_4/gamma
.:,�2!RMSprop/velocity/batchnorm_4/beta
2:0���2RMSprop/velocity/dense_1/kernel
*:(�2RMSprop/velocity/dense_1/bias
/:-�2"RMSprop/velocity/batchnorm_7/gamma
.:,�2!RMSprop/velocity/batchnorm_7/beta
2:0	�2!RMSprop/velocity/out_layer/kernel
+:)2RMSprop/velocity/out_layer/bias
�B�
"__inference__update_step_xla_40779gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40784gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40789gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40794gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40799gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40804gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40809gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40814gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40819gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40824gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40829gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40834gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40839gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40844gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40849gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40854gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40859gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40864gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40869gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_40874gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
?__inference_DCNN_layer_call_and_return_conditional_losses_40410�"()*+23<=>?MNVW`abcwx������G�D
=�:
0�-
conv2d_1_input���������00
p

 
� ",�)
"�
tensor_0���������
� �
?__inference_DCNN_layer_call_and_return_conditional_losses_40499�"()*+23<=>?MNVW`abcwx������G�D
=�:
0�-
conv2d_1_input���������00
p 

 
� ",�)
"�
tensor_0���������
� �
$__inference_DCNN_layer_call_fn_40560�"()*+23<=>?MNVW`abcwx������G�D
=�:
0�-
conv2d_1_input���������00
p

 
� "!�
unknown����������
$__inference_DCNN_layer_call_fn_40621�"()*+23<=>?MNVW`abcwx������G�D
=�:
0�-
conv2d_1_input���������00
p 

 
� "!�
unknown����������
"__inference__update_step_xla_40779~x�u
n�k
!�
gradient@
<�9	%�"
�@
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40784f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40789f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`���Ă�?
� "
 �
"__inference__update_step_xla_40794f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`���Ă�?
� "
 �
"__inference__update_step_xla_40799~x�u
n�k
!�
gradient@@
<�9	%�"
�@@
�
p
` VariableSpec 
`��Ă�?
� "
 �
"__inference__update_step_xla_40804f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`��Ă�?
� "
 �
"__inference__update_step_xla_40809f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40814f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40819�z�w
p�m
"�
gradient@�
=�:	&�#
�@�
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40824hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40829�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40834hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40839hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�ܞ���?
� "
 �
"__inference__update_step_xla_40844hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�ព��?
� "
 �
"__inference__update_step_xla_40849tn�k
d�a
�
gradient���
7�4	 �
����
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40854hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40859hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40864hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_40869pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`���Ă�?
� "
 �
"__inference__update_step_xla_40874f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`���Ă�?
� "
 �
 __inference__wrapped_model_39962�"()*+23<=>?MNVW`abcwx������?�<
5�2
0�-
conv2d_1_input���������00
� "5�2
0
	out_layer#� 
	out_layer����������
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40938�()*+Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_40956�()*+Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
+__inference_batchnorm_1_layer_call_fn_40907�()*+Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
+__inference_batchnorm_1_layer_call_fn_40920�()*+Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41020�<=>?Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_41038�<=>?Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
+__inference_batchnorm_2_layer_call_fn_40989�<=>?Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
+__inference_batchnorm_2_layer_call_fn_41002�<=>?Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41149�`abcR�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_41167�`abcR�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
+__inference_batchnorm_4_layer_call_fn_41118�`abcR�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
+__inference_batchnorm_4_layer_call_fn_41131�`abcR�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41285s����8�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
tensor_0����������
� �
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_41305s����8�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
tensor_0����������
� �
+__inference_batchnorm_7_layer_call_fn_41238h����8�5
.�+
!�
inputs����������
p

 
� ""�
unknown�����������
+__inference_batchnorm_7_layer_call_fn_41251h����8�5
.�+
!�
inputs����������
p 

 
� ""�
unknown�����������
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40894s7�4
-�*
(�%
inputs���������00
� "4�1
*�'
tensor_0���������00@
� �
(__inference_conv2d_1_layer_call_fn_40883h7�4
-�*
(�%
inputs���������00
� ")�&
unknown���������00@�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40976s237�4
-�*
(�%
inputs���������00@
� "4�1
*�'
tensor_0���������00@
� �
(__inference_conv2d_2_layer_call_fn_40965h237�4
-�*
(�%
inputs���������00@
� ")�&
unknown���������00@�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_41085tMN7�4
-�*
(�%
inputs���������00@
� "5�2
+�(
tensor_0���������00�
� �
(__inference_conv2d_3_layer_call_fn_41074iMN7�4
-�*
(�%
inputs���������00@
� "*�'
unknown���������00��
C__inference_conv2d_4_layer_call_and_return_conditional_losses_41105uVW8�5
.�+
)�&
inputs���������00�
� "5�2
+�(
tensor_0���������00�
� �
(__inference_conv2d_4_layer_call_fn_41094jVW8�5
.�+
)�&
inputs���������00�
� "*�'
unknown���������00��
B__inference_dense_1_layer_call_and_return_conditional_losses_41225fwx1�.
'�$
"�
inputs�����������
� "-�*
#� 
tensor_0����������
� �
'__inference_dense_1_layer_call_fn_41214[wx1�.
'�$
"�
inputs�����������
� ""�
unknown�����������
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060s;�8
1�.
(�%
inputs���������00@
p
� "4�1
*�'
tensor_0���������00@
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_41065s;�8
1�.
(�%
inputs���������00@
p 
� "4�1
*�'
tensor_0���������00@
� �
)__inference_dropout_1_layer_call_fn_41043h;�8
1�.
(�%
inputs���������00@
p
� ")�&
unknown���������00@�
)__inference_dropout_1_layer_call_fn_41048h;�8
1�.
(�%
inputs���������00@
p 
� ")�&
unknown���������00@�
D__inference_dropout_2_layer_call_and_return_conditional_losses_41189u<�9
2�/
)�&
inputs���������00�
p
� "5�2
+�(
tensor_0���������00�
� �
D__inference_dropout_2_layer_call_and_return_conditional_losses_41194u<�9
2�/
)�&
inputs���������00�
p 
� "5�2
+�(
tensor_0���������00�
� �
)__inference_dropout_2_layer_call_fn_41172j<�9
2�/
)�&
inputs���������00�
p
� "*�'
unknown���������00��
)__inference_dropout_2_layer_call_fn_41177j<�9
2�/
)�&
inputs���������00�
p 
� "*�'
unknown���������00��
D__inference_dropout_4_layer_call_and_return_conditional_losses_41327e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
D__inference_dropout_4_layer_call_and_return_conditional_losses_41332e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
)__inference_dropout_4_layer_call_fn_41310Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
)__inference_dropout_4_layer_call_fn_41315Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
B__inference_flatten_layer_call_and_return_conditional_losses_41205j8�5
.�+
)�&
inputs���������00�
� ".�+
$�!
tensor_0�����������
� �
'__inference_flatten_layer_call_fn_41199_8�5
.�+
)�&
inputs���������00�
� "#� 
unknown������������
D__inference_out_layer_layer_call_and_return_conditional_losses_41352f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
)__inference_out_layer_layer_call_fn_41341[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
#__inference_signature_wrapper_40774�"()*+23<=>?MNVW`abcwx������Q�N
� 
G�D
B
conv2d_1_input0�-
conv2d_1_input���������00"5�2
0
	out_layer#� 
	out_layer���������