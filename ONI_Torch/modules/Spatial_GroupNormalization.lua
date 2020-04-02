local GN,parent = torch.class('nn.Spatial_GroupNormalization', 'nn.Module')

function GN:__init(nFeature, GroupNumber,affine, eps, momentum)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call SpatialBatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   if affine ~=nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
 --  self.affine=false
   self.GroupNumber=GroupNumber or 32
   if self.GroupNumber> nFeature then
      self.GroupNumber=nFeature
   end
   print('GroupNumber:'..self.GroupNumber)
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1

   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
      self.gradWeight = torch.Tensor(nFeature)
      self.gradBias = torch.Tensor(nFeature)
      self:reset()
   end
end

function GN:reset()
   self.weight:uniform():fill(1)
   self.bias:zero()
end

function GN:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.output:resizeAs(input)
      self.buffer2 = self.buffer2 or input.new()
      self.centered = self.centered or input.new()
      self.std = self.std or input.new()
      self.normalized = self.normalized or input.new()
      self.normalized:resizeAs(input)
      self.gradInput:resizeAs(input)
      -- calculate mean over mini-batch, over feature-maps
      if not input:isContiguous() then
          self._input = self._input or input.new()
          self._input:typeAs(input):resizeAs(input):copy(input)
          input = self._input
       end

      self.centered:resizeAs(input)
      local N_perGroup=nFeature/self.GroupNumber
      local input_reshape=input:reshape(nBatch,self.GroupNumber,N_perGroup,iH,iW)
      local in_folded = input_reshape:view(nBatch, self.GroupNumber, N_perGroup*iH * iW)
      self.buffer:mean(in_folded, 3)

      -- subtract mean
      self.centered:add(input_reshape, -1, self.buffer:view(nBatch,self.GroupNumber,1,1,1):expandAs(input_reshape))                  -- x - E(x)

      -- calculate standard deviation over mini-batch
      self.buffer:resizeAs(self.centered):copy(self.centered):cmul(self.buffer)          -- [x - E(x)]^2
      local buf_folded = self.buffer:view(nBatch,self.GroupNumber,N_perGroup*iH*iW)
      self.std:mean(buf_folded, 3)
      self.std:add(self.eps):sqrt():pow(-1)      -- 1 / E([x - E(x)]^2)

      -- divide standard-deviation + eps
      self.output:cmul(self.centered, self.std:view(nBatch,self.GroupNumber,1,1,1):expandAs(self.centered))
      self.output=self.output:reshape(nBatch,nFeature,iH,iW)
                                                                                  self.normalized:copy(self.output)

  -- print(self.output:size())
  -- print(self.normalized:size())
   if self.affine then
      -- multiply with gamma and add beta
      self.output:cmul(self.weight:view(1,nFeature,1,1):expandAs(self.output))
     self.output:add(self.bias:view(1,nFeature,1,1):expandAs(self.output))
   end
   collectgarbage()
   return self.output
end

function GN:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   --assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)


      if not gradOutput:isContiguous() then
          self._gradOutput = self._gradOutput or input.new()
          self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
          gradOutput = self._gradOutput
       end
   local N_perGroup=nFeature/self.GroupNumber
   self.buffer:resizeAs(gradOutput):copy(gradOutput) 
    -- print(gradOutput)
    if self.affine then
   -- print('----------back  scaling-----------')
      self.buffer:cmul(self.weight:view(1,nFeature,1,1):expandAs(gradOutput))
   end
     -- print(gradOutput)
      self.gradInput:cmul(self.normalized,self.buffer)
      self.gradInput=self.gradInput:reshape(nBatch,self.GroupNumber,N_perGroup,iH,iW)
      local gi_folded = self.gradInput:view(nBatch, self.GroupNumber, N_perGroup*iH * iW)
      self.buffer2:mean(gi_folded, 3)
   
      self.gradInput:copy(-self.normalized):cmul(self.buffer2:view(nBatch,self.GroupNumber,1,1,1):expandAs(self.gradInput))
   
      self.buffer=self.buffer:reshape(nBatch,self.GroupNumber,N_perGroup,iH,iW)
      local go_folded=self.buffer:view(nBatch, self.GroupNumber,N_perGroup* iH*iW)
      self.buffer2:mean(go_folded, 3)
       self.gradInput:add(self.buffer):add(-1, self.buffer2:view(nBatch,self.GroupNumber,1,1,1):expandAs(self.buffer))
       self.gradInput:cmul(self.std:view(nBatch,self.GroupNumber,1,1,1):expandAs(self.gradInput))
       self.gradInput=self.gradInput:reshape(nBatch,nFeature,iH,iW)
  return self.gradInput
end

function GN:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      self.buffer:resizeAs(self.normalized):copy(self.normalized)
      self.buffer = self.buffer:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
      self.buffer2:sum(self.buffer:sum(3), 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer2)

      self.buffer2:sum(gradOutput:view(nBatch, nFeature, iH*iW):sum(3), 1)
      self.gradBias:add(scale, self.buffer2) -- sum over mini-batch
   end
end
