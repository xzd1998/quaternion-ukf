Search.setIndex({docnames:["estimator","estimator.data","estimator.sandbox","includeme","index","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["estimator.rst","estimator.data.rst","estimator.sandbox.rst","includeme.rst","index.rst","modules.rst"],objects:{"":{estimator:[0,0,0,"-"]},"estimator.data":{datamaker:[1,0,0,"-"],datasource:[1,0,0,"-"],datastore:[1,0,0,"-"],trainer:[1,0,0,"-"],trajectoryplanner:[1,0,0,"-"],utilities:[1,0,0,"-"]},"estimator.data.datamaker":{DataMaker:[1,1,1,""]},"estimator.data.datasource":{DataSource:[1,1,1,""]},"estimator.data.datasource.DataSource":{angles:[1,2,1,""]},"estimator.data.datastore":{DataStore:[1,1,1,""]},"estimator.data.trainer":{Trainer:[1,1,1,""]},"estimator.data.trainer.Trainer":{IMU_COEFFICIENTS:[1,3,1,""],IMU_INTERCEPTS:[1,3,1,""],clip_data:[1,2,1,""],train_acc:[1,2,1,""],train_vel:[1,2,1,""]},"estimator.data.trajectoryplanner":{RoundTripPlanner:[1,1,1,""],SimplePlanner:[1,1,1,""],StationaryPlanner:[1,1,1,""],TrajectoryPlanner:[1,1,1,""]},"estimator.data.trajectoryplanner.TrajectoryPlanner":{decrementer:[1,2,1,""],get_calculator:[1,2,1,""],incrementer:[1,2,1,""]},"estimator.data.utilities":{accs_to_roll_pitch:[1,4,1,""],angles_to_rots_xyz:[1,4,1,""],angles_to_rots_zyx:[1,4,1,""],make_angles_continuous:[1,4,1,""],moving_average:[1,4,1,""],normalize_vectors:[1,4,1,""],plot_rowwise_data:[1,4,1,""],rots_to_accs:[1,4,1,""],rots_to_angles_xyz:[1,4,1,""],rots_to_angles_zyx:[1,4,1,""],rots_to_vectors:[1,4,1,""],rots_to_vels:[1,4,1,""],vectors_to_rots:[1,4,1,""]},"estimator.quaternion_integrator":{QuaternionIntegrator:[0,1,1,""]},"estimator.quaternion_integrator.QuaternionIntegrator":{estimate_state:[0,2,1,""],state_dof:[0,3,1,""]},"estimator.quaternions":{Quaternions:[0,1,1,""],ZeroQuaternionException:[0,5,1,""]},"estimator.quaternions.Quaternions":{array:[0,2,1,""],epsilon:[0,3,1,""],find_q_mean:[0,2,1,""],from_list:[0,2,1,""],from_quaternions:[0,2,1,""],from_vectors:[0,2,1,""],inverse:[0,2,1,""],ndim:[0,3,1,""],q_multiply:[0,2,1,""],rotate_vector:[0,2,1,""],to_rotation_matrix:[0,2,1,""],to_vectors:[0,2,1,""]},"estimator.quaternionukf":{QuaternionUkf:[0,1,1,""]},"estimator.quaternionukf.QuaternionUkf":{estimate_state:[0,2,1,""],g_vector:[0,3,1,""],state_dof:[0,3,1,""]},"estimator.roll_pitch_calculator":{RollPitchCalculator:[0,1,1,""]},"estimator.roll_pitch_calculator.RollPitchCalculator":{N_DIM:[0,3,1,""],estimate_state:[0,2,1,""]},"estimator.sandbox":{quaternionukf3:[2,0,0,"-"],vectorukf3:[2,0,0,"-"],vectorukf6:[2,0,0,"-"]},"estimator.sandbox.quaternionukf3":{QuaternionUkf3:[2,1,1,""]},"estimator.sandbox.quaternionukf3.QuaternionUkf3":{estimate_state:[2,2,1,""],g_vector:[2,3,1,""],state_dof:[2,3,1,""]},"estimator.sandbox.vectorukf3":{VectorUkf3:[2,1,1,""]},"estimator.sandbox.vectorukf3.VectorUkf3":{estimate_state:[2,2,1,""],g_vector:[2,3,1,""],state_dof:[2,3,1,""]},"estimator.sandbox.vectorukf6":{VectorUkf6:[2,1,1,""]},"estimator.sandbox.vectorukf6.VectorUkf6":{estimate_state:[2,2,1,""],g_vector:[2,3,1,""]},"estimator.state_estimator":{StateEstimator:[0,1,1,""]},"estimator.state_estimator.StateEstimator":{acc_data:[0,2,1,""],angles:[0,2,1,""],estimate_state:[0,2,1,""],imu_data:[0,2,1,""],num_data:[0,2,1,""],plot_comparison:[0,2,1,""],state_dof:[0,2,1,""],ts_imu:[0,2,1,""],vel_data:[0,2,1,""]},"estimator.vector_integrator":{VectorIntegrator:[0,1,1,""]},"estimator.vector_integrator.VectorIntegrator":{estimate_state:[0,2,1,""]},estimator:{constants:[0,0,0,"-"],data:[1,0,0,"-"],quaternion_integrator:[0,0,0,"-"],quaternions:[0,0,0,"-"],quaternionukf:[0,0,0,"-"],roll_pitch_calculator:[0,0,0,"-"],sandbox:[2,0,0,"-"],state_estimator:[0,0,0,"-"],vector_integrator:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:exception"},terms:{"abstract":0,"class":[0,1,2],"final":0,"function":[0,2],"import":1,"null":[0,2],"return":0,"static":[0,1],The:1,There:3,Uses:[0,2],abc:0,abl:1,abov:1,acc:1,acc_calcul:1,acc_data:[0,1],acc_magnitud:1,acceleromet:[0,1],acceleromt:0,accs_to_roll_pitch:1,added:1,advantag:0,after:[0,1,2],all:[0,2],alpha:[0,2],angl:[0,1],angles_to_rots_xyz:1,angles_to_rots_zyx:1,angular:1,arrai:[0,1,2],associ:0,base:[0,1,2,4],becaus:1,beta:[0,2],both:[0,1],bound:1,broadcast:0,calibr:1,call:[0,2],captur:0,classmethod:0,clip:1,clip_data:1,code:3,coef_det:1,coeff:1,coeffici:1,coeffient:1,come:1,comparison:0,constant:[4,5],construct:0,content:[4,5],convent:1,convert:0,creat:1,data:[0,2,4,5],data_label:1,data_to_clip:1,datamak:[0,4,5],dataset:[1,3],dataset_numb:1,datasourc:[0,4,5],datastor:[0,4,5],decrement:1,defin:0,degre:0,depend:0,descent:0,differ:1,drift_stddev:1,drone:0,durat:1,either:1,element:0,epsilon:0,equip:0,estim:4,estimate_st:[0,2],exampl:0,except:0,exist:1,expect:1,filter:[0,2],find:0,find_q_mean:0,first:1,form:1,freedom:0,from:[0,1,3],from_list:0,from_quaternion:0,from_vector:0,g_vector:[0,2],get_calcul:1,gradient:0,graviti:1,ground:1,guarante:1,guess:0,gyro:[0,1],gyroscop:1,has:1,have:0,histori:[0,2],implement:0,imu:[0,1],imu_coeffici:1,imu_data:[0,1],imu_intercept:1,includ:[],increment:1,index:4,initi:0,integr:0,inter:1,intercept:1,interfac:[0,1],interpol:1,invers:0,iter:0,ith:1,kappa:[0,2],keep:0,length:0,line:1,linear:1,list:[0,1],load:1,made:1,mai:0,make:0,make_angles_continu:1,match:1,matric:[0,1],matrix:[0,1],max:0,mean:0,meant:0,member:1,messag:0,model:1,modul:[4,5],moving_averag:1,multipli:[0,1],must:[0,2],n_dim:0,ndim:0,nofilt:0,nois:1,noise_stddev:1,none:1,normalize_vector:1,note:1,num_data:0,number:[0,3],numpi:0,object:[0,1],orient:[0,4],otherwis:0,overriden:0,packag:[],page:4,param:0,parent:[0,1],path_to_data:1,pitch:[0,1],plan:1,planner:1,plot:0,plot_comparison:0,plot_rowwise_data:1,pre:1,project:0,properti:[0,2],provid:[0,2],python3:3,q_mean:0,q_multipli:0,quaternion:[4,5],quaternion_integr:[4,5],quaternionintegr:0,quaternionukf3:[0,4,5],quaternionukf:5,raw:[0,1],readm:[],real:1,refer:1,reference_data:1,regress:1,respect:1,result:[0,1],roll:[0,1],roll_pitch_calcul:[4,5],rollpitchcalcul:0,rot:1,rotat:[0,1],rotate_vector:0,rots_est:0,rots_to_acc:1,rots_to_angles_xyz:1,rots_to_angles_zyx:1,rots_to_vector:1,rots_to_vel:1,rots_vicon:[0,1],roundtripplann:1,row:1,rst:[],run:3,same:1,sampl:3,sandbox:[0,4,5],search:4,simpleplann:1,singleton:0,six:0,slightli:1,solv:1,sourc:[0,1,2],specifi:1,state:[0,2],state_dof:[0,2],state_estim:[2,4,5],stateestim:[0,2],stationaryplann:1,store:1,submodul:[4,5],subpackag:[4,5],take:0,termin:3,test:1,thi:[0,2],three:[0,1,3],time:[0,1],to_rotation_matrix:0,to_vector:0,togeth:0,track:[0,4],train:1,train_acc:1,train_vel:1,trainer:[0,4,5],trajectori:1,trajectoryplann:[0,4,5],transform:1,truth:[0,1],ts_imu:[0,1],ts_vicon:[0,1],tupl:[0,1],two:[0,1],ukf:[0,4],unless:1,usag:4,using:[0,1],util:[0,4,5],variou:0,vector:[0,1],vector_integr:[4,5],vectorintegr:0,vectors_to_rot:1,vectorukf3:[0,4,5],vectorukf6:[0,4,5],vel_data:[0,1],veloc:1,versu:0,vicon:1,where:3,which:[0,1],y_label:1,yaw:[0,1],zeroquaternionexcept:0},titles:["estimator","estimator.data","estimator.sandbox","Quaternion-Based UKF for Orientation Tracking","Welcome to QuaternionUkf\u2019s documentation!","estimator"],titleterms:{base:3,constant:0,content:[0,1,2],data:1,datamak:1,datasourc:1,datastor:1,document:4,estim:[0,1,2,5],indic:4,modul:[0,1,2],orient:3,packag:[],quaternion:[0,3],quaternion_integr:0,quaternionukf3:2,quaternionukf:[0,4],readm:[],roll_pitch_calcul:0,sandbox:2,state_estim:0,submodul:[0,1,2],subpackag:0,tabl:4,track:3,trainer:1,trajectoryplann:1,ukf:3,usag:3,util:1,vector_integr:0,vectorukf3:2,vectorukf6:2,welcom:4}})