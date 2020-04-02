
local nn = require 'nn'
require 'cunn'
require 'cudnn'
require '../../modules/Spatial_Weight_BN'
local Convolution = cudnn.Spatial_OrthReg_OI
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

local function createModel(opt)
   local depth = opt.depth
   local iChannels

   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(cudnn.Spatial_Weight_BN(nInputPlane,n,true, opt.N_scale,3,3,stride,stride,1,1))
      return nn.Sequential()
            :add(s)
         :add(ReLU(true))
   end

   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
     assert((depth - 2) % 3 == 0, 'depth should be 3n+2')
      local n = (depth - 2) / 3
      local k=opt.widen_factor
      print(' | ResNet-' .. depth .. ' CIFAR-10')
      iChannels = 16*k

      model:add(cudnn.Spatial_Weight_BN(3,16*k,true,opt.N_scale,3,3,1,1,1,1))
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

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.Spatial_Weight_BN')
   ConvInit('nn.SpatialConvolution')
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
