
local nn = require 'nn'
require 'cunn'
require 'cudnn'
require '../../modules/Spatial_Weight_DBN_PowerIter_F2'

local Convolution = cudnn.Spatial_Weight_DBN_PowerIter_F2
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
--local SBatchNorm = nn.SpatialBatchNormalization
--local CudnnSBatchNorm = cudnn.SpatialBatchNormalization

local function Dropout()
    return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function WhiteNoise()
    return nn.WhiteNoise_add_mul(opt.dropout,opt.dropout)
end

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane,opt.m_perGroup,opt.nIter, true,opt.N_scale, 1, 1, stride, stride))
           -- :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,opt.m_perGroup,opt.nIter,true, opt.N_scale,3,3,stride,stride,1,1))
      --s:add(SBatchNorm(n))
      if opt.dropout > 0 then
           --s:add(Dropout())
           s:add(WhiteNoise())
      end

      s:add(ReLU(true))
      s:add(Convolution(n,n,opt.m_perGroup,opt.nIter,true, opt.N_scale,3,3,1,1,1,1))

      --s:add(CudnnSBatchNorm(n))
      if opt.dropout > 0 then
           --s:add(Dropout())
           s:add(WhiteNoise())
      end

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end


   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      local k=opt.widen_factor
      iChannels = 16*k
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16*k,opt.m_perGroup,opt.nIter,true, opt.N_scale,3,3,1,1,1,1))
      --model:add(SBatchNorm(16*k))
      if opt.dropout > 0 then
           --model:add(Dropout())
           model:add(WhiteNoise())
      end
      model:add(ReLU(true))
      model:add(layer(basicblock, 16*k, n))
      model:add(layer(basicblock, 32*k, n, 2))
      model:add(layer(basicblock, 64*k, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64*k):setNumInputDims(3))
      model:add(nn.Linear(64*k, opt.num_classes))

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name, value)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(value)
         v.bias:zero()
      end
   end

   --ConvInit('cudnn.Spatial_Weight_DBN_PowerIter_F2')
   --ConvInit('nn.SpatialConvolution')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'cudnn_deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   --model:get(1).gradInput = nil

   return model
end

return createModel(opt)
