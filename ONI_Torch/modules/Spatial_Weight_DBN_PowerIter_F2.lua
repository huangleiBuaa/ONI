local Spatial_Weight_DBN_PowerIter_F2, parent =
    torch.class('cudnn.Spatial_Weight_DBN_PowerIter_F2', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

function Spatial_Weight_DBN_PowerIter_F2:__init(nInputPlane, nOutputPlane,m_perGroup, nIter, flag_adjustScale, N_scale,
                            kW, kH, dW, dH, padW, padH)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
   
    self.W = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradW = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
  
   self.inputDim=nInputPlane*self.kH*self.kW
   if flag_adjustScale ~= nil then
      self.flag_adjustScale= flag_adjustScale
     else
       self.flag_adjustScale= false
    end
    if N_scale ~= nil then
        self.N_scale= N_scale
    else
        self.N_scale= 1
    end

   self.g=torch.Tensor(nOutputPlane):fill(self.N_scale)
   
    if nIter ~= nil then
      self.nIter= nIter
     else
       self.nIter= 3
    end

   if m_perGroup~=nil then
      if nOutputPlane > self.inputDim then
      self.m_perGroup = m_perGroup > self.inputDim and self.inputDim  or m_perGroup
      else
       self.m_perGroup=m_perGroup > nOutputPlane and nOutputPlane  or m_perGroup
      end
  else
     self.m_perGroup =   nOutputPlane > self.inputDim and self.inputDim or nOutputPlane
   end

   print("m_perGroup:"..self.m_perGroup..'-----nIter:'..self.nIter)


   self.eps=1e-7
    self.n_groups=torch.floor((nOutputPlane-1)/self.m_perGroup)+1

    local length = self.m_perGroup
    self.eye_ngroup = torch.eye(length):cuda()

    length = nOutputPlane - (self.n_groups - 1) * self.m_perGroup
    self.eye_ngroup_last = torch.eye(length):cuda()
   
   -- print(self.eye_ngroup)
   -- print(self.eye_ngroup_last)
     print('-----------scaling:'..self.N_scale)
      if self.flag_adjustScale then
         print('-----------using adjust scale-------------')
         self.gradG=torch.Tensor(nOutputPlane)
      end


    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self.isTraining=true

    self:reset_centered_orthogonal()


------------------for debug------------------

   -- self.debug=true
   if self.debug then
    self.correlate=torch.Tensor(nOutputPlane,nOutputPlane)
    self.conditionNumber_input={}
    self.conditionNumber_gradOutput={}
    self.maxEig_input={}
    self.maxEig_gradOutput={}
    self.count=0
    self.interval=interval or 10
    self.eig_input={}
    self.eig_gradOutput={}
    self.epcilo=10^-35
   end
end

function Spatial_Weight_DBN_PowerIter_F2:reset_centered_orthogonal()
   function init_perGroup(weight_perGroup,groupId)
      local n_output=weight_perGroup:size(1)
      local n_input=weight_perGroup:size(2)
     local scale=weight_perGroup.new()
      local centered = weight_perGroup.new()
      local sigma=weight_perGroup.new()
      local I_Matrix=torch.eye(n_output)
      local W_perGroup=weight_perGroup.new()
      W_perGroup:resizeAs(weight_perGroup)
      local buffer=weight_perGroup.new()
      buffer:mean(weight_perGroup, 2) 
      centered:add(weight_perGroup, -1, buffer:expandAs(weight_perGroup))

       ----------------------calcualte the projection matrix----------------------
      sigma:resize(n_output,n_output)
     -- sigma:addmm(0,sigma,1/n_input,centered,centered:t()) --
      sigma:addmm(0,sigma,1,centered,centered:t()) --for Debug
     -- sigma:add(self.eps, I_Matrix)
      -----------------------PowerIter_F2------------- 
      local trace=torch.norm(sigma)    --using F2 norm
      local sigma_norm=sigma/trace
      local X=I_Matrix:clone()
      local nIter=10 
      for i=1, nIter do
          X=(3*X-X*X*X*sigma_norm)/2
      end
    
      local whiten_matrix=X/torch.sqrt(trace)
      W_perGroup:mm(whiten_matrix,centered)
      local validate_Matrix=W_perGroup*W_perGroup:t()
      local diag_sum=torch.diag(validate_Matrix):sum()
      local non_diag_sum_abs=validate_Matrix:abs():sum()-diag_sum
      local nDim=validate_Matrix:size(1)
      print('Init:--------diag_mean:'..(diag_sum/nDim) ..'---non_diag_mean:'..(non_diag_sum_abs/nDim))
      return W_perGroup
  end      
    


 ---------------init main function----------------------
   self.weight = self.weight:view(self.nOutputPlane,self.nInputPlane*self.kH*self.kW)
   local n_output=self.weight:size(1)
   local n_input=self.weight:size(2)
   local  data=torch.randn(self.weight:size(1),self.weight:size(2))

      for i=1,self.n_groups do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.weight[{{start_index,end_index},{}}]=init_perGroup(data[{{start_index,end_index},{}}],i)
      end
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)

end


-- if you change the configuration of the module manually, call this
function Spatial_Weight_DBN_PowerIter_F2:resetWeightDescriptors(desc)
    -- for compatibility
    self.groups = self.groups or 1
    assert(cudnn.typemap[torch.typename(self.W)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end

    self.WDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.W)],
         filterDimA = desc or
            {self.nOutputPlane/self.groups,
             self.nInputPlane/self.groups,
             self.kH, self.kW}
       }
    )

    return self
end

function Spatial_Weight_DBN_PowerIter_F2:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
       self.fastest_mode = mode
       self.iDesc = nil
    end
    return self
end

function Spatial_Weight_DBN_PowerIter_F2:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    return self
end

function Spatial_Weight_DBN_PowerIter_F2:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function Spatial_Weight_DBN_PowerIter_F2:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end


function Spatial_Weight_DBN_PowerIter_F2:checkInputChanged(input)
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.groups = self.groups or 1
    if not self.WDesc then self:resetWeightDescriptors() end
    if not self.WDesc then error "Weights not assigned!" end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert(self.nInputPlane == input:size(2),
              'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
       return true
    end
    return false
end

function Spatial_Weight_DBN_PowerIter_F2:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if Spatial_Weight_DBN_PowerIter_F2.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = cudnn.toDescriptor(input_slice)
        -- create conv descriptor
        self.padH, self.padW = self.padH or 0, self.padW or 0
        -- those needed to calculate hash
        self.pad = {self.padH, self.padW}
        self.stride = {self.dH, self.dW}

        self.convDescData = { padA = self.pad,
             filterStrideA = self.stride,
             upscaleA = {1,1},
             dataType = cudnn.configmap(torch.type(self.W))
        }

        self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.WDesc[0], 4, oSize:data())
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.W_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
   return self
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end



-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewGradWeight(self)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewGradWeight(self)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end
-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewW(self)
   self.W = self.W:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
end

local function unviewW(self)
   self.W = self.W:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewGradW(self)
   if self.gradW and self.gradW:dim() > 0 then
      self.gradW = self.gradW:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewGradW(self)
   if self.gradW and self.gradW:dim() > 0 then
      self.gradW = self.gradW:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end



function Spatial_Weight_DBN_PowerIter_F2:updateOutput(input)

----used for the group eigen composition---------------------------

   function updateOutput_perGroup(weight_perGroup,groupId)
      
      local n_output=weight_perGroup:size(1)
      local n_input=weight_perGroup:size(2)
     local scale=weight_perGroup.new()
      local centered = weight_perGroup.new()
      local sigma=weight_perGroup.new()
      local set_X={}
      local I_Matrix
      if groupId ~= self.n_groups then
         I_Matrix=self.eye_ngroup
      else
         I_Matrix=self.eye_ngroup_last
      end

      self.W_perGroup=self.W_perGroup or input.new()
      self.W_perGroup:resizeAs(weight_perGroup)
      self.buffer:mean(weight_perGroup, 2) 
      centered:add(weight_perGroup, -1, self.buffer:expandAs(weight_perGroup))

       ----------------------calcualte the projection matrix----------------------
      sigma:resize(n_output,n_output)
     -- sigma:addmm(0,sigma,1/n_input,centered,centered:t()) --
      sigma:addmm(0,sigma,1,centered,centered:t()) --for Debug
      --sigma:add(self.eps, I_Matrix)
     

      -----------------------PowerIter_F2------------- 
      
      local trace=torch.norm(sigma)    --using F2 norm
      local sigma_norm=sigma/trace
      local X=I_Matrix:clone()
     
     -- print(sigma:size())
     -- print(X:size())
      for i=1, self.nIter do
          X=(3*X-X*X*X*sigma_norm)/2
          table.insert(set_X, X:clone())
      end
    
      local whiten_matrix=X/torch.sqrt(trace)
       
      self.W_perGroup:mm(whiten_matrix,centered)
     -- print(self.W_perGroup*self.W_perGroup:t())

         ----------------record the results of per groupt--------------
      table.insert(self.centereds, centered)
      table.insert(self.sigmas, sigma)
      table.insert(self.whiten_matrixs, whiten_matrix)
      table.insert(self.set_Xs, set_X)
  
      return self.W_perGroup
  end      
    


 ---------------update main function----------------------



    input = makeContiguous(self, input)
   
   -----------------------start of the debug-----------------
    if self.train and self.debug and (self.count % self.interval)==0 then
    -------------------------------for the input--------------
       local nBatch = input:size(1)
       local nDim=input:size(2)
       local iH=input:size(3)
       local iW=input:size(4)
       self.input_temp= self.input_temp or input.new() 
       self.input_temp=input:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()
      self.correlate:resize(nDim,nDim)
      self.correlate:addmm(0,self.correlate, 1/self.input_temp:size(1),self.input_temp:t(), self.input_temp)
      local _,buffer,_=torch.svd(self.correlate:clone():float()) --SVD Decompositon for singular value
      --print(buffer) 
       table.insert(self.eig_input,buffer:clone())
       buffer:add(self.epcilo)
       local maxEig=torch.max(buffer)
       local conditionNumber=torch.abs(maxEig/torch.min(buffer))
       print('input conditionNumber='..conditionNumber..'----maxEig:'..maxEig)
       self.conditionNumber_input[#self.conditionNumber_input + 1]=conditionNumber
       self.maxEig_input[#self.maxEig_input + 1]=maxEig
    end 
   -----------------------end of the debug-----------------


    self:createIODescriptors(input)

if self.isTraining then 
    -----------------------------transform----------------------
   viewWeight(self)
   viewW(self)
   local n_output=self.weight:size(1)
   local n_input=self.weight:size(2)

    self.sigmas={}
   self.set_Xs={}
   self.whiten_matrixs={}
   self.centereds={}

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer_1 = self.buffer_1 or input.new()
   self.buffer_2 = self.buffer_2 or input.new()

   self.output=self.output or input.new()
   self.W=self.W or input.new()
   self.W:resizeAs(self.weight)


      for i=1,self.n_groups do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.W[{{start_index,end_index},{}}]=updateOutput_perGroup(self.weight[{{start_index,end_index},{}}],i)
      end
   --if self.flag_adjustScale then
       self.W_hat=self.W_hat or input.new()
       self.W_hat:resizeAs(self.W):copy(self.W)
       self.W:cmul(self.g:view(n_output,1):expandAs(self.W))
   --end
 
   unviewW(self)
   unviewWeight(self)
end



------------------------------------------------cudnn excute-----------------------

    local finder = find.get()
    local fwdAlgo = finder:forwardAlgorithm(self, { self.iDesc[0], self.input_slice, self.WDesc[0],
                                                    self.W, self.convDesc[0], self.oDesc[0], self.output_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.iDesc[0], input:data() + g*self.input_offset,
                    self.WDesc[0], self.W:data() + g*self.W_offset,
                    self.convDesc[0], fwdAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end
    collectgarbage()

    return self.output
end

function Spatial_Weight_DBN_PowerIter_F2:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    assert(gradOutput:dim() == input:dim()-1 or gradOutput:dim() == input:dim()
              or (gradOutput:dim()==5 and input:dim()==4), 'Wrong gradOutput dimensions');
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    local bwdDataAlgo = finder:backwardDataAlgorithm(self, { self.WDesc[0], self.W, self.oDesc[0],
                                                             self.output_slice, self.convDesc[0], self.iDesc[0], self.input_slice })
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0,self.groups - 1 do
        checkedCall(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.WDesc[0], self.W:data() + g*self.W_offset,
                    self.oDesc[0], gradOutput:data() + g*self.output_offset,
                    self.convDesc[0],
                    bwdDataAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.iDesc[0], self.gradInput:data() + g*self.input_offset)
    end
    return self.gradInput
end

function Spatial_Weight_DBN_PowerIter_F2:accGradParameters(input, gradOutput, scale)


   function Matrix_Pow3(input)
      local b=torch.mm(input, input)
      return torch.mm(b,input)
   end

 ---------------------------------------------------------

function updateAccGradParameters_perGroup(gradW_perGroup, groupId)
    local n_output=gradW_perGroup:size(1)
     local n_input=gradW_perGroup:size(2)
     

     local  sigma=self.sigmas[groupId]
     local  centered=self.centereds[groupId]
     local  whiten_matrix=self.whiten_matrixs[groupId]
     local  set_X=self.set_Xs[groupId]
     local trace=torch.norm(sigma)
     local sigma_norm=sigma/trace
     local I_Matrix

     self.gradWeight_perGroup=self.gradWeight_perGroup or gradW_perGroup.new()
     self.gradWeight_perGroup:resizeAs(gradW_perGroup)
     self.dC=self.dC or gradW_perGroup.new()
     self.dA=self.dA or gradW_perGroup.new()
     self.dSigma=self.dSigma or gradW_perGroup.new()
     self.dXN=self.dXN or gradW_perGroup.new()
     self.f=self.f or gradW_perGroup.new()
     self.dC:resizeAs(whiten_matrix)
     self.dA:resizeAs(whiten_matrix)

   
     self.dC:mm(gradW_perGroup, centered:t())
     self.dXN=self.dC/torch.sqrt(trace)
     if groupId ~= self.n_groups then
         I_Matrix=self.eye_ngroup
     else
        I_Matrix=self.eye_ngroup_last
    end



    local P3
    if self.nIter ==1 then 
       P3=I_Matrix
    else
       P3=Matrix_Pow3(set_X[self.nIter-1])
    end
   
    self.dA:mm(P3:t(),self.dXN)
    local dX_kPlus=self.dXN

    for i=self.nIter-1, 1, -1 do 
      ----calculate dL/dx_k+1--------------
       local X=set_X[i]
       local tmp1=dX_kPlus*(X*X*sigma_norm):t()
       local tmp2=X:t()* dX_kPlus *(X*sigma_norm):t()
       local tmp3=(X*X):t()* dX_kPlus * sigma_norm:t()
       local dX_k= 3*dX_kPlus/2-(tmp1+tmp2+tmp3)/2
     ------update dA--------------
       if  i ~= 1  then
         local    X_before=set_X[i-1]
         local tmp=Matrix_Pow3(X_before)
         self.dA:add(tmp:t()*dX_k)
         dX_kPlus=dX_k
       else
         self.dA:add(dX_k)
       end

    end
   
     self.dA=-self.dA/2
     
     --local s1=torch.trace(self.dA:t()*sigma)
     local s1=torch.cmul(self.dA, sigma):sum() --more efficient implementation
     --local s2=torch.trace(self.dC:t()*set_X[self.nIter]) 
     local s2=torch.cmul(self.dC, set_X[self.nIter]):sum()
     
     --self.dSigma=-(s1/(trace^2)+s2/(2*trace^(3/2)))*I_Matrix
     self.dSigma=-(s1/(trace^2)+s2/(2*trace^(3/2)))*(sigma/trace)
     self.dSigma=self.dSigma+self.dA/trace
     local dSigma_sym=(self.dSigma+self.dSigma:t())/2


     self.f:mean(gradW_perGroup, 2)
     local d_mean=gradW_perGroup-self.f:expandAs(gradW_perGroup) 
     self.buffer:resizeAs(gradW_perGroup) 
    self.buffer:mm(whiten_matrix,d_mean) 
     self.gradWeight_perGroup=(2)*dSigma_sym*centered
     self.gradWeight_perGroup=self.gradWeight_perGroup+self.buffer  
    
       
      
      return self.gradWeight_perGroup
  end     
 

 -------------------------------------main function of accGrad------------







    self.scaleT = self.scaleT or self.W.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.W) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    input, gradOutput = makeContiguous(self, input, gradOutput)
   
   -------------------------start of debug------------------------------
    if self.train and self.debug  then
      if (self.count % self.interval) ==0 then
      
       local nBatch = gradOutput:size(1)
       local nDim=gradOutput:size(2)
       local iH=gradOutput:size(3)
       local iW=gradOutput:size(4)
       self.input_temp=gradOutput:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()
    self.correlate:resize(nDim,nDim)
    -------------------------------for the input--------------
      self.correlate:addmm(0,self.correlate, self.input_temp:size(1),self.input_temp:t(), self.input_temp)
      local _,buffer,_=torch.svd(self.correlate:clone():float()) --SVD Decompositon for singular value
      --print(buffer) 
       table.insert(self.eig_gradOutput,buffer:clone())
       buffer:add(self.epcilo)
       local maxEig=torch.max(buffer)
       local conditionNumber=torch.abs(maxEig/torch.min(buffer))
       print('gradOutput conditionNumber='..conditionNumber..'----maxEig:'..maxEig)
       self.conditionNumber_gradOutput[#self.conditionNumber_gradOutput + 1]=conditionNumber
       self.maxEig_gradOutput[#self.maxEig_gradOutput + 1]=maxEig
    end 
    self.count=self.count+1 
   end 
----------------------end of debug---------------------
   
    self:createIODescriptors(input)
   self.gradW:fill(0)

    local finder = find.get()
    local bwdFilterAlgo = finder:backwardFilterAlgorithm(self, { self.iDesc[0], self.input_slice, self.oDesc[0],
                                                               self.output_slice, self.convDesc[0], self.WDesc[0], self.W})

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end

    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        -- gradWeight
       checkedCall(self,'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                   self.scaleT:data(),
                   self.iDesc[0], input:data() + g*self.input_offset,
                   self.oDesc[0], gradOutput:data() + g*self.output_offset,
                   self.convDesc[0],
                   bwdFilterAlgo,
                   extraBuffer, extraBufferSize,
                   cudnn.scalar(input, 1),
                   self.WDesc[0], self.gradW:data() + g*self.W_offset);
    end




-----------------------------transform--------------------------

    
   viewWeight(self)
   viewW(self)
   viewGradWeight(self)
   viewGradW(self)
    local n_output=self.weight:size(1)
   
   if self.flag_adjustScale then 
      self.W_hat:cmul(self.gradW)
      self.gradG:sum(self.W_hat,2)
   end
    
   self.gradW:cmul(self.g:view(n_output,1):expandAs(self.gradWeight))

   for i=1,self.n_groups do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.gradWeight[{{start_index,end_index},{}}]=updateAccGradParameters_perGroup(self.gradW[{{start_index,end_index},{}}],i)
    end




   unviewWeight(self)
   unviewW(self)
   unviewGradWeight(self)
   unviewGradW(self)



    return self.gradOutput
end

function Spatial_Weight_DBN_PowerIter_F2:clearDesc()
    self.WDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.oSize = nil
    self.scaleT = nil
    return self
end

function Spatial_Weight_DBN_PowerIter_F2:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function Spatial_Weight_DBN_PowerIter_F2:parameters()
   if self.flag_adjustScale then
     --print('retun g and weight')
     return {self.weight, self.g}, {self.gradWeight, self.gradG}
   else
     --print('retun weight')
     return {self.weight}, {self.gradWeight}
  end
end


function Spatial_Weight_DBN_PowerIter_F2:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput', 'input_slice', 'output_slice')
   return nn.Module.clearState(self)
end

function Spatial_Weight_DBN_PowerIter_F2:endTraining()

   function updateOutput_perGroup(weight_perGroup,groupId)
      local n_output=weight_perGroup:size(1)
      local n_input=weight_perGroup:size(2)
     local scale=weight_perGroup.new()
      local centered = weight_perGroup.new()
      local sigma=weight_perGroup.new()
      local I_Matrix
      if groupId ~= self.n_groups then
         I_Matrix=self.eye_ngroup
      else
         I_Matrix=self.eye_ngroup_last
      end
      self.W_perGroup=self.W_perGroup or input.new()
      self.W_perGroup:resizeAs(weight_perGroup)
      self.buffer:mean(weight_perGroup, 2) 
      centered:add(weight_perGroup, -1, self.buffer:expandAs(weight_perGroup))
    ----------------------calcualte the projection matrix----------------------
      sigma:resize(n_output,n_output)
     -- sigma:addmm(0,sigma,1/n_input,centered,centered:t()) --
      sigma:addmm(0,sigma,1,centered,centered:t()) --for Debug
      --sigma:add(self.eps, I_Matrix)
      -----------------------PowerIter_F2------------- 
      local trace=torch.norm(sigma)    --using F2 norm
      local sigma_norm=sigma/trace
      local X=I_Matrix:clone()
      for i=1, self.nIter do
          X=(3*X-X*X*X*sigma_norm)/2
      end
      local whiten_matrix=X/torch.sqrt(trace)
      self.W_perGroup:mm(whiten_matrix,centered)
      print(self.W_perGroup*self.W_perGroup:t())
      return self.W_perGroup
  end      
    



  ------------------------main funciton-------------------
   viewWeight(self)
   viewW(self)
   local n_output=self.weight:size(1)
   local n_input=self.weight:size(2)

      for i=1,self.n_groups do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.W[{{start_index,end_index},{}}]=updateOutput_perGroup(self.weight[{{start_index,end_index},{}}],i)
      end
   --if self.flag_adjustScale then
       self.W_hat=self.W_hat or input.new()
       self.W_hat:resizeAs(self.W):copy(self.W)
       self.W:cmul(self.g:view(n_output,1):expandAs(self.W))
   --end
 
   unviewW(self)
   unviewWeight(self)
   self.isTraining=false
   ------------------clear buffer-----------
--    self.weight:set()
   print('----------------clear buff-----------')
  self.buffer:set()
 --   self.gradWeight:set()
--    self.gradW:set()
--    self.W_hat:set()
--    self.W_perGroup:set()
 --   self.gradInput:set()
 --   self.W_perGroup:set()
 --   self.gradWeight_perGroup:set()
end
