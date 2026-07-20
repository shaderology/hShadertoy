#include <interpolate.h>
#include <matrix.h>
#include <random.h>
#include <imx.h>
#include <imx_filter.h>
#define AT_elemnum      _bound_idx
#define AT_ix   _bound_gidx
#define AT_iy   _bound_gidy
#define AT_ixy  (int2)(_bound_gidx, _bound_gidy)
#define AT_res  (_RUNOVER_LAYER.stat->resolution)
#define AT_xres (_RUNOVER_LAYER.stat->resolution.x)
#define AT_yres (_RUNOVER_LAYER.stat->resolution.y)
#define AT_P_image      _bound_P_image
#define AT_P_pixel      _bound_P_pixel
#define AT_P_texture    _bound_P_texture
#define AT_P_world      (imageToWorld(_RUNOVER_LAYER.stat, _bound_P_image))
#define AT_P    AT_P_image
#define AT_dPdx_image   ((float2)(_RUNOVER_LAYER.stat->buffer_to_image.x,0))
#define AT_dPdx_pixel   ((float2)(_RUNOVER_LAYER.stat->buffer_to_pixel.x,0))
#define AT_dPdx_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,0))
#define AT_dPdx_world   (_RUNOVER_LAYER.stat->buffer_to_image.x * _RUNOVER_LAYER.stat->image_to_world.lo.lo.xyz)
#define AT_dPdx AT_dPdx_image
#define AT_dPdy_image   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_image.y))
#define AT_dPdy_pixel   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_pixel.y))
#define AT_dPdy_texture ((float2)(0, 1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdy_world   (_RUNOVER_LAYER.stat->buffer_to_image.y * _RUNOVER_LAYER.stat->image_to_world.lo.hi.xyz)
#define AT_dPdy AT_dPdy_image
#define AT_dPdxy_image (_RUNOVER_LAYER.stat->buffer_to_image.xy)
#define AT_dPdxy_pixel (_RUNOVER_LAYER.stat->buffer_to_pixel.xy)
#define AT_dPdxy_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdxy AT_dPdxy_image
#define AT_tilesize     _bound_tilesize
#define AT_Time _bound_time
#define AT_iDate        _bound_iDate
#define AT_iFrame       _bound_iFrame
#define AT_iFrameRate   _bound_iFrameRate
#define AT_iMouse       _bound_iMouse
#define AT_iTime        _bound_iTime
#ifdef HAS_size_ref
#define AT_size_ref_data        _bound_size_ref
#else
#define AT_size_ref_data        0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bound       1
#else
#define AT_size_ref_bound       0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_stat        ((global IMX_Stat * restrict) _bound_size_ref_stat_void)
#else
#define AT_size_ref_stat        0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_layer       &_bound_size_ref_layer
#else
#define AT_size_ref_layer       0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_border      _bound_size_ref_border
#else
#define AT_size_ref_border      IMX_WRAP
#endif
#ifdef HAS_size_ref
#define AT_size_ref_storage     _bound_size_ref_storage
#else
#define AT_size_ref_storage     FLOAT32
#endif
#ifdef HAS_size_ref
#define AT_size_ref_channels    _bound_size_ref_channels
#else
#define AT_size_ref_channels    4
#endif
#ifdef HAS_size_ref
#define AT_size_ref_tuplesize   _bound_size_ref_channels
#else
#define AT_size_ref_tuplesize   4
#endif
#ifdef HAS_size_ref
#define AT_size_ref_xres        _bound_size_ref_layer.stat->resolution.x
#else
#define AT_size_ref_xres        1
#endif
#ifdef HAS_size_ref
#define AT_size_ref_yres        _bound_size_ref_layer.stat->resolution.y
#else
#define AT_size_ref_yres        1
#endif
#ifdef HAS_size_ref
#define AT_size_ref_res convert_float2(_bound_size_ref_layer.stat->resolution)
#else
#define AT_size_ref_res (float2)(1)
#endif
#define CONSTANT1(s) CONSTANT_ ## s
#define CONSTANT_(s) CONSTANT1(s)
#ifdef CONSTANT_size_ref
#define size_ref_args2 CONSTANT_(_bound_size_ref_storage), _bound_size_ref_channels
#else
#define size_ref_args2 _bound_size_ref_storage, _bound_size_ref_channels
#endif
#define size_ref_args3 _bound_size_ref_border, size_ref_args2
#ifdef HAS_size_ref
#define AT_size_ref_bufferIndex(_xy_)   bufferIndexF4(&_bound_size_ref_layer, _xy_, size_ref_args3)
#else
#define AT_size_ref_bufferIndex(_xy_)   _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSample(_xy_)  bufferSampleF4(&_bound_size_ref_layer, _xy_, size_ref_args3)
#else
#define AT_size_ref_bufferSample(_xy_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageNearest(_xy_)  bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(imageToBuffer(AT_size_ref_stat, _xy_) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_imageNearest(_xy_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSample(_xy_)   bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_imageSample(_xy_)   _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldNearest(_xyz_) bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(imageToBuffer(AT_size_ref_stat, worldToImage(AT_size_ref_stat, _xyz_)) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_worldNearest(_xyz_) _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldSample(_xyz_)  bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, worldToImage(AT_size_ref_stat, _xyz_)), size_ref_args3)
#else
#define AT_size_ref_worldSample(_xyz_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureNearest(_xy_)        bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(textureToBuffer(AT_size_ref_stat, _xy_) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_textureNearest(_xy_)        _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSample(_xy_) bufferSampleF4(&_bound_size_ref_layer, textureToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_textureSample(_xy_) _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_1(_xy_)     bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_1(_xy_)     _bound_size_ref
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref     _bufferIndexLinF4(&_bound_size_ref_layer, _bound_idx, size_ref_args2)
#else
#define AT_size_ref     bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _bound_P_image), size_ref_args3)
#endif
#else
#define AT_size_ref     _bound_size_ref
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref_dCdx        dCdxF4aligned(&_bound_size_ref_layer, (int2)(_bound_gidx, _bound_gidy), size_ref_args3)
#else
#define AT_size_ref_dCdx        dCdxF4(&_bound_size_ref_layer, _bound_P_image, size_ref_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_size_ref_dCdx        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_dCdx_1(_xy_)        dCdxF4(&_bound_size_ref_layer, _xy_, size_ref_args3, &_RUNOVER_LAYER)
#else
#define AT_size_ref_dCdx_1(_xy_)        ((float4)0)
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref_dCdy        dCdyF4aligned(&_bound_size_ref_layer, (int2)(_bound_gidx, _bound_gidy), size_ref_args3)
#else
#define AT_size_ref_dCdy        dCdyF4(&_bound_size_ref_layer, _bound_P_image, size_ref_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_size_ref_dCdy        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_dCdy_1(_xy_)        dCdyF4(&_bound_size_ref_layer, _xy_, size_ref_args3, &_RUNOVER_LAYER)
#else
#define AT_size_ref_dCdy_1(_xy_)        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSampleRect(_xy_, _dxy_)       bufferSampleRectF4(&_bound_size_ref_layer, _xy_, _dxy_, size_ref_args3)
#else
#define AT_size_ref_bufferSampleRect(_xy_, _dxy_)       _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSampleRectClip(_xy_, _dxy_)   bufferSampleRectClipF4(&_bound_size_ref_layer, _xy_, _dxy_, size_ref_args2)
#else
#define AT_size_ref_bufferSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(bufferToImage(AT_size_ref_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y)), _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSampleRect(_xy_, _dxy_)        AT_size_ref_bufferSampleRect(imageToBuffer(AT_size_ref_stat, _xy_), AT_size_ref_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_size_ref_imageSampleRect(_xy_, _dxy_)        _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSampleRectClip(_xy_, _dxy_)    AT_size_ref_bufferSampleRectClip(imageToBuffer(AT_size_ref_stat, _xy_), AT_size_ref_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_size_ref_imageSampleRectClip(_xy_, _dxy_)    constImageSampleRectClip(_xy_, _dxy_, _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSampleRect(_xy_, _dxy_)      AT_size_ref_bufferSampleRect(textureToBuffer(AT_size_ref_stat, _xy_), (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y) * (_dxy_))
#else
#define AT_size_ref_textureSampleRect(_xy_, _dxy_)      _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSampleRectClip(_xy_, _dxy_)  AT_size_ref_bufferSampleRectClip(textureToBuffer(AT_size_ref_stat, _xy_), (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y) * (_dxy_))
#else
#define AT_size_ref_textureSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_size_ref_stat, textureToBuffer(AT_size_ref_stat, _xy_)), _dxy_ * ((float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y)) * AT_size_ref_stat->buffer_to_image.lo, _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToImage(_xy_) (bufferToImage(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToImage(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageToBuffer(_xy_) (imageToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_imageToBuffer(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToPixel(_xy_) (bufferToPixel(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToPixel(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_pixelToBuffer(_xy_) (pixelToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_pixelToBuffer(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToTexture(_xy_)       (bufferToTexture(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToTexture(_xy_)       (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureToBuffer(_xy_)       (textureToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_textureToBuffer(_xy_)       (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageToWorld(_xy_)  (imageToWorld(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_imageToWorld(_xy_)  ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_size_ref
#define AT_size_ref_image3ToWorld(_xyz_)        (image3ToWorld(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_image3ToWorld(_xyz_)        (_xyz_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldToImage(_xyz_) (worldToImage(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_worldToImage(_xyz_) ((_xyz_).xy)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldToImage3(_xyz_)        (worldToImage3(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_worldToImage3(_xyz_)        (_xyz_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_data       _bound_fragCoord
#else
#define AT_fragCoord_data       0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bound      1
#else
#define AT_fragCoord_bound      0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_stat       ((global IMX_Stat * restrict) _bound_fragCoord_stat_void)
#else
#define AT_fragCoord_stat       0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_layer      &_bound_fragCoord_layer
#else
#define AT_fragCoord_layer      0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_border     _bound_fragCoord_border
#else
#define AT_fragCoord_border     IMX_WRAP
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_storage    _bound_fragCoord_storage
#else
#define AT_fragCoord_storage    FLOAT32
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_channels   _bound_fragCoord_channels
#else
#define AT_fragCoord_channels   4
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_tuplesize  _bound_fragCoord_channels
#else
#define AT_fragCoord_tuplesize  4
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_xres       _bound_fragCoord_layer.stat->resolution.x
#else
#define AT_fragCoord_xres       1
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_yres       _bound_fragCoord_layer.stat->resolution.y
#else
#define AT_fragCoord_yres       1
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_res        convert_float2(_bound_fragCoord_layer.stat->resolution)
#else
#define AT_fragCoord_res        (float2)(1)
#endif
#ifdef CONSTANT_fragCoord
#define fragCoord_args2 CONSTANT_(_bound_fragCoord_storage), _bound_fragCoord_channels
#else
#define fragCoord_args2 _bound_fragCoord_storage, _bound_fragCoord_channels
#endif
#define fragCoord_args3 _bound_fragCoord_border, fragCoord_args2
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferIndex(_xy_)  bufferIndexF2(&_bound_fragCoord_layer, _xy_, fragCoord_args3)
#else
#define AT_fragCoord_bufferIndex(_xy_)  _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSample(_xy_) bufferSampleF2(&_bound_fragCoord_layer, _xy_, fragCoord_args3)
#else
#define AT_fragCoord_bufferSample(_xy_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageNearest(_xy_) bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(imageToBuffer(AT_fragCoord_stat, _xy_) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_imageNearest(_xy_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSample(_xy_)  bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_imageSample(_xy_)  _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldNearest(_xyz_)        bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(imageToBuffer(AT_fragCoord_stat, worldToImage(AT_fragCoord_stat, _xyz_)) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_worldNearest(_xyz_)        _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldSample(_xyz_) bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, worldToImage(AT_fragCoord_stat, _xyz_)), fragCoord_args3)
#else
#define AT_fragCoord_worldSample(_xyz_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureNearest(_xy_)       bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(textureToBuffer(AT_fragCoord_stat, _xy_) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_textureNearest(_xy_)       _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSample(_xy_)        bufferSampleF2(&_bound_fragCoord_layer, textureToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_textureSample(_xy_)        _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_1(_xy_)    bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_1(_xy_)    _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord    _bufferIndexLinF2(&_bound_fragCoord_layer, _bound_idx, fragCoord_args2)
#else
#define AT_fragCoord    bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _bound_P_image), fragCoord_args3)
#endif
#else
#define AT_fragCoord    _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord_dCdx       dCdxF4aligned(&_bound_fragCoord_layer, (int2)(_bound_gidx, _bound_gidy), fragCoord_args3).xy
#else
#define AT_fragCoord_dCdx       dCdxF4(&_bound_fragCoord_layer, _bound_P_image, fragCoord_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_fragCoord_dCdx       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_dCdx_1(_xy_)       dCdxF4(&_bound_fragCoord_layer, _xy_, fragCoord_args3, &_RUNOVER_LAYER).xy
#else
#define AT_fragCoord_dCdx_1(_xy_)       ((float2)0)
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord_dCdy       dCdyF4aligned(&_bound_fragCoord_layer, (int2)(_bound_gidx, _bound_gidy), fragCoord_args3).xy
#else
#define AT_fragCoord_dCdy       dCdyF4(&_bound_fragCoord_layer, _bound_P_image, fragCoord_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_fragCoord_dCdy       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_dCdy_1(_xy_)       dCdyF4(&_bound_fragCoord_layer, _xy_, fragCoord_args3, &_RUNOVER_LAYER).xy
#else
#define AT_fragCoord_dCdy_1(_xy_)       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_fragCoord_layer, _xy_, _dxy_, fragCoord_args3).xy
#else
#define AT_fragCoord_bufferSampleRect(_xy_, _dxy_)      _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_fragCoord_layer, _xy_, _dxy_, fragCoord_args2).xy
#else
#define AT_fragCoord_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_fragCoord_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y)), _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSampleRect(_xy_, _dxy_)       AT_fragCoord_bufferSampleRect(imageToBuffer(AT_fragCoord_stat, _xy_), AT_fragCoord_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_fragCoord_imageSampleRect(_xy_, _dxy_)       _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSampleRectClip(_xy_, _dxy_)   AT_fragCoord_bufferSampleRectClip(imageToBuffer(AT_fragCoord_stat, _xy_), AT_fragCoord_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_fragCoord_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSampleRect(_xy_, _dxy_)     AT_fragCoord_bufferSampleRect(textureToBuffer(AT_fragCoord_stat, _xy_), (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y) * (_dxy_))
#else
#define AT_fragCoord_textureSampleRect(_xy_, _dxy_)     _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSampleRectClip(_xy_, _dxy_) AT_fragCoord_bufferSampleRectClip(textureToBuffer(AT_fragCoord_stat, _xy_), (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y) * (_dxy_))
#else
#define AT_fragCoord_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_fragCoord_stat, textureToBuffer(AT_fragCoord_stat, _xy_)), _dxy_ * ((float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y)) * AT_fragCoord_stat->buffer_to_image.lo, _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToImage(_xy_)        (bufferToImage(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageToBuffer(_xy_)        (imageToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToPixel(_xy_)        (bufferToPixel(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_pixelToBuffer(_xy_)        (pixelToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToTexture(_xy_)      (bufferToTexture(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureToBuffer(_xy_)      (textureToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageToWorld(_xy_) (imageToWorld(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_image3ToWorld(_xyz_)       (image3ToWorld(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldToImage(_xyz_)        (worldToImage(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldToImage3(_xyz_)       (worldToImage3(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_data       _bound_iChannel0
#else
#define AT_iChannel0_data       0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bound      1
#else
#define AT_iChannel0_bound      0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_stat       ((global IMX_Stat * restrict) _bound_iChannel0_stat_void)
#else
#define AT_iChannel0_stat       0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_layer      &_bound_iChannel0_layer
#else
#define AT_iChannel0_layer      0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_border     _bound_iChannel0_border
#else
#define AT_iChannel0_border     IMX_WRAP
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_storage    _bound_iChannel0_storage
#else
#define AT_iChannel0_storage    FLOAT32
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_channels   _bound_iChannel0_channels
#else
#define AT_iChannel0_channels   4
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_tuplesize  _bound_iChannel0_channels
#else
#define AT_iChannel0_tuplesize  4
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_xres       _bound_iChannel0_layer.stat->resolution.x
#else
#define AT_iChannel0_xres       1
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_yres       _bound_iChannel0_layer.stat->resolution.y
#else
#define AT_iChannel0_yres       1
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_res        convert_float2(_bound_iChannel0_layer.stat->resolution)
#else
#define AT_iChannel0_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel0
#define iChannel0_args2 CONSTANT_(_bound_iChannel0_storage), _bound_iChannel0_channels
#else
#define iChannel0_args2 _bound_iChannel0_storage, _bound_iChannel0_channels
#endif
#define iChannel0_args3 _bound_iChannel0_border, iChannel0_args2
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferIndex(_xy_)  _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferSample(_xy_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel0_stat, _xy_) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_imageNearest(_xy_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_imageSample(_xy_)  _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel0_stat, worldToImage(AT_iChannel0_stat, _xyz_)) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_worldNearest(_xyz_)        _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, worldToImage(AT_iChannel0_stat, _xyz_)), iChannel0_args3)
#else
#define AT_iChannel0_worldSample(_xyz_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel0_stat, _xy_) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_textureNearest(_xy_)       _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel0_layer, textureToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_textureSample(_xy_)        _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_1(_xy_)    bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_1(_xy_)    _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0    _bufferIndexLinF4(&_bound_iChannel0_layer, _bound_idx, iChannel0_args2)
#else
#define AT_iChannel0    bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _bound_P_image), iChannel0_args3)
#endif
#else
#define AT_iChannel0    _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0_dCdx       dCdxF4aligned(&_bound_iChannel0_layer, (int2)(_bound_gidx, _bound_gidy), iChannel0_args3)
#else
#define AT_iChannel0_dCdx       dCdxF4(&_bound_iChannel0_layer, _bound_P_image, iChannel0_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel0_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel0_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0_dCdy       dCdyF4aligned(&_bound_iChannel0_layer, (int2)(_bound_gidx, _bound_gidy), iChannel0_args3)
#else
#define AT_iChannel0_dCdy       dCdyF4(&_bound_iChannel0_layer, _bound_P_image, iChannel0_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel0_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel0_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel0_layer, _xy_, _dxy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel0_layer, _xy_, _dxy_, iChannel0_args2)
#else
#define AT_iChannel0_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel0_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y)), _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSampleRect(_xy_, _dxy_)       AT_iChannel0_bufferSampleRect(imageToBuffer(AT_iChannel0_stat, _xy_), AT_iChannel0_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel0_imageSampleRect(_xy_, _dxy_)       _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel0_bufferSampleRectClip(imageToBuffer(AT_iChannel0_stat, _xy_), AT_iChannel0_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel0_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSampleRect(_xy_, _dxy_)     AT_iChannel0_bufferSampleRect(textureToBuffer(AT_iChannel0_stat, _xy_), (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel0_textureSampleRect(_xy_, _dxy_)     _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSampleRectClip(_xy_, _dxy_) AT_iChannel0_bufferSampleRectClip(textureToBuffer(AT_iChannel0_stat, _xy_), (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel0_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel0_stat, textureToBuffer(AT_iChannel0_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y)) * AT_iChannel0_stat->buffer_to_image.lo, _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToImage(_xy_)        (bufferToImage(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageToWorld(_xy_) (imageToWorld(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldToImage(_xyz_)        (worldToImage(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_data       _bound_iChannel1
#else
#define AT_iChannel1_data       0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bound      1
#else
#define AT_iChannel1_bound      0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_stat       ((global IMX_Stat * restrict) _bound_iChannel1_stat_void)
#else
#define AT_iChannel1_stat       0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_layer      &_bound_iChannel1_layer
#else
#define AT_iChannel1_layer      0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_border     _bound_iChannel1_border
#else
#define AT_iChannel1_border     IMX_WRAP
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_storage    _bound_iChannel1_storage
#else
#define AT_iChannel1_storage    FLOAT32
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_channels   _bound_iChannel1_channels
#else
#define AT_iChannel1_channels   4
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_tuplesize  _bound_iChannel1_channels
#else
#define AT_iChannel1_tuplesize  4
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_xres       _bound_iChannel1_layer.stat->resolution.x
#else
#define AT_iChannel1_xres       1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_yres       _bound_iChannel1_layer.stat->resolution.y
#else
#define AT_iChannel1_yres       1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_res        convert_float2(_bound_iChannel1_layer.stat->resolution)
#else
#define AT_iChannel1_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel1
#define iChannel1_args2 CONSTANT_(_bound_iChannel1_storage), _bound_iChannel1_channels
#else
#define iChannel1_args2 _bound_iChannel1_storage, _bound_iChannel1_channels
#endif
#define iChannel1_args3 _bound_iChannel1_border, iChannel1_args2
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferIndex(_xy_)  _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferSample(_xy_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel1_stat, _xy_) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_imageNearest(_xy_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_imageSample(_xy_)  _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel1_stat, worldToImage(AT_iChannel1_stat, _xyz_)) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_worldNearest(_xyz_)        _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, worldToImage(AT_iChannel1_stat, _xyz_)), iChannel1_args3)
#else
#define AT_iChannel1_worldSample(_xyz_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel1_stat, _xy_) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_textureNearest(_xy_)       _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel1_layer, textureToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_textureSample(_xy_)        _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_1(_xy_)    bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_1(_xy_)    _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1    _bufferIndexLinF4(&_bound_iChannel1_layer, _bound_idx, iChannel1_args2)
#else
#define AT_iChannel1    bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _bound_P_image), iChannel1_args3)
#endif
#else
#define AT_iChannel1    _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1_dCdx       dCdxF4aligned(&_bound_iChannel1_layer, (int2)(_bound_gidx, _bound_gidy), iChannel1_args3)
#else
#define AT_iChannel1_dCdx       dCdxF4(&_bound_iChannel1_layer, _bound_P_image, iChannel1_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel1_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel1_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1_dCdy       dCdyF4aligned(&_bound_iChannel1_layer, (int2)(_bound_gidx, _bound_gidy), iChannel1_args3)
#else
#define AT_iChannel1_dCdy       dCdyF4(&_bound_iChannel1_layer, _bound_P_image, iChannel1_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel1_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel1_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel1_layer, _xy_, _dxy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel1_layer, _xy_, _dxy_, iChannel1_args2)
#else
#define AT_iChannel1_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel1_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y)), _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSampleRect(_xy_, _dxy_)       AT_iChannel1_bufferSampleRect(imageToBuffer(AT_iChannel1_stat, _xy_), AT_iChannel1_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel1_imageSampleRect(_xy_, _dxy_)       _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel1_bufferSampleRectClip(imageToBuffer(AT_iChannel1_stat, _xy_), AT_iChannel1_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel1_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSampleRect(_xy_, _dxy_)     AT_iChannel1_bufferSampleRect(textureToBuffer(AT_iChannel1_stat, _xy_), (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel1_textureSampleRect(_xy_, _dxy_)     _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSampleRectClip(_xy_, _dxy_) AT_iChannel1_bufferSampleRectClip(textureToBuffer(AT_iChannel1_stat, _xy_), (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel1_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel1_stat, textureToBuffer(AT_iChannel1_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y)) * AT_iChannel1_stat->buffer_to_image.lo, _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToImage(_xy_)        (bufferToImage(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageToWorld(_xy_) (imageToWorld(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldToImage(_xyz_)        (worldToImage(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_data       _bound_iChannel2
#else
#define AT_iChannel2_data       0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bound      1
#else
#define AT_iChannel2_bound      0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_stat       ((global IMX_Stat * restrict) _bound_iChannel2_stat_void)
#else
#define AT_iChannel2_stat       0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_layer      &_bound_iChannel2_layer
#else
#define AT_iChannel2_layer      0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_border     _bound_iChannel2_border
#else
#define AT_iChannel2_border     IMX_WRAP
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_storage    _bound_iChannel2_storage
#else
#define AT_iChannel2_storage    FLOAT32
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_channels   _bound_iChannel2_channels
#else
#define AT_iChannel2_channels   4
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_tuplesize  _bound_iChannel2_channels
#else
#define AT_iChannel2_tuplesize  4
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_xres       _bound_iChannel2_layer.stat->resolution.x
#else
#define AT_iChannel2_xres       1
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_yres       _bound_iChannel2_layer.stat->resolution.y
#else
#define AT_iChannel2_yres       1
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_res        convert_float2(_bound_iChannel2_layer.stat->resolution)
#else
#define AT_iChannel2_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel2
#define iChannel2_args2 CONSTANT_(_bound_iChannel2_storage), _bound_iChannel2_channels
#else
#define iChannel2_args2 _bound_iChannel2_storage, _bound_iChannel2_channels
#endif
#define iChannel2_args3 _bound_iChannel2_border, iChannel2_args2
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferIndex(_xy_)  _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferSample(_xy_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel2_stat, _xy_) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_imageNearest(_xy_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_imageSample(_xy_)  _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel2_stat, worldToImage(AT_iChannel2_stat, _xyz_)) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_worldNearest(_xyz_)        _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, worldToImage(AT_iChannel2_stat, _xyz_)), iChannel2_args3)
#else
#define AT_iChannel2_worldSample(_xyz_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel2_stat, _xy_) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_textureNearest(_xy_)       _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel2_layer, textureToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_textureSample(_xy_)        _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_1(_xy_)    bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_1(_xy_)    _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2    _bufferIndexLinF4(&_bound_iChannel2_layer, _bound_idx, iChannel2_args2)
#else
#define AT_iChannel2    bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _bound_P_image), iChannel2_args3)
#endif
#else
#define AT_iChannel2    _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2_dCdx       dCdxF4aligned(&_bound_iChannel2_layer, (int2)(_bound_gidx, _bound_gidy), iChannel2_args3)
#else
#define AT_iChannel2_dCdx       dCdxF4(&_bound_iChannel2_layer, _bound_P_image, iChannel2_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel2_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel2_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2_dCdy       dCdyF4aligned(&_bound_iChannel2_layer, (int2)(_bound_gidx, _bound_gidy), iChannel2_args3)
#else
#define AT_iChannel2_dCdy       dCdyF4(&_bound_iChannel2_layer, _bound_P_image, iChannel2_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel2_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel2_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel2_layer, _xy_, _dxy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel2_layer, _xy_, _dxy_, iChannel2_args2)
#else
#define AT_iChannel2_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel2_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y)), _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSampleRect(_xy_, _dxy_)       AT_iChannel2_bufferSampleRect(imageToBuffer(AT_iChannel2_stat, _xy_), AT_iChannel2_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel2_imageSampleRect(_xy_, _dxy_)       _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel2_bufferSampleRectClip(imageToBuffer(AT_iChannel2_stat, _xy_), AT_iChannel2_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel2_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSampleRect(_xy_, _dxy_)     AT_iChannel2_bufferSampleRect(textureToBuffer(AT_iChannel2_stat, _xy_), (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel2_textureSampleRect(_xy_, _dxy_)     _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSampleRectClip(_xy_, _dxy_) AT_iChannel2_bufferSampleRectClip(textureToBuffer(AT_iChannel2_stat, _xy_), (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel2_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel2_stat, textureToBuffer(AT_iChannel2_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y)) * AT_iChannel2_stat->buffer_to_image.lo, _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToImage(_xy_)        (bufferToImage(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageToWorld(_xy_) (imageToWorld(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldToImage(_xyz_)        (worldToImage(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_data       _bound_iChannel3
#else
#define AT_iChannel3_data       0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bound      1
#else
#define AT_iChannel3_bound      0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_stat       ((global IMX_Stat * restrict) _bound_iChannel3_stat_void)
#else
#define AT_iChannel3_stat       0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_layer      &_bound_iChannel3_layer
#else
#define AT_iChannel3_layer      0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_border     _bound_iChannel3_border
#else
#define AT_iChannel3_border     IMX_WRAP
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_storage    _bound_iChannel3_storage
#else
#define AT_iChannel3_storage    FLOAT32
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_channels   _bound_iChannel3_channels
#else
#define AT_iChannel3_channels   4
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_tuplesize  _bound_iChannel3_channels
#else
#define AT_iChannel3_tuplesize  4
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_xres       _bound_iChannel3_layer.stat->resolution.x
#else
#define AT_iChannel3_xres       1
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_yres       _bound_iChannel3_layer.stat->resolution.y
#else
#define AT_iChannel3_yres       1
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_res        convert_float2(_bound_iChannel3_layer.stat->resolution)
#else
#define AT_iChannel3_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel3
#define iChannel3_args2 CONSTANT_(_bound_iChannel3_storage), _bound_iChannel3_channels
#else
#define iChannel3_args2 _bound_iChannel3_storage, _bound_iChannel3_channels
#endif
#define iChannel3_args3 _bound_iChannel3_border, iChannel3_args2
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferIndex(_xy_)  _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferSample(_xy_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel3_stat, _xy_) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_imageNearest(_xy_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_imageSample(_xy_)  _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel3_stat, worldToImage(AT_iChannel3_stat, _xyz_)) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_worldNearest(_xyz_)        _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, worldToImage(AT_iChannel3_stat, _xyz_)), iChannel3_args3)
#else
#define AT_iChannel3_worldSample(_xyz_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel3_stat, _xy_) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_textureNearest(_xy_)       _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel3_layer, textureToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_textureSample(_xy_)        _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_1(_xy_)    bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_1(_xy_)    _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3    _bufferIndexLinF4(&_bound_iChannel3_layer, _bound_idx, iChannel3_args2)
#else
#define AT_iChannel3    bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _bound_P_image), iChannel3_args3)
#endif
#else
#define AT_iChannel3    _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3_dCdx       dCdxF4aligned(&_bound_iChannel3_layer, (int2)(_bound_gidx, _bound_gidy), iChannel3_args3)
#else
#define AT_iChannel3_dCdx       dCdxF4(&_bound_iChannel3_layer, _bound_P_image, iChannel3_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel3_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel3_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3_dCdy       dCdyF4aligned(&_bound_iChannel3_layer, (int2)(_bound_gidx, _bound_gidy), iChannel3_args3)
#else
#define AT_iChannel3_dCdy       dCdyF4(&_bound_iChannel3_layer, _bound_P_image, iChannel3_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel3_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel3_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel3_layer, _xy_, _dxy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel3_layer, _xy_, _dxy_, iChannel3_args2)
#else
#define AT_iChannel3_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel3_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y)), _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSampleRect(_xy_, _dxy_)       AT_iChannel3_bufferSampleRect(imageToBuffer(AT_iChannel3_stat, _xy_), AT_iChannel3_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel3_imageSampleRect(_xy_, _dxy_)       _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel3_bufferSampleRectClip(imageToBuffer(AT_iChannel3_stat, _xy_), AT_iChannel3_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel3_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSampleRect(_xy_, _dxy_)     AT_iChannel3_bufferSampleRect(textureToBuffer(AT_iChannel3_stat, _xy_), (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel3_textureSampleRect(_xy_, _dxy_)     _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSampleRectClip(_xy_, _dxy_) AT_iChannel3_bufferSampleRectClip(textureToBuffer(AT_iChannel3_stat, _xy_), (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel3_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel3_stat, textureToBuffer(AT_iChannel3_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y)) * AT_iChannel3_stat->buffer_to_image.lo, _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToImage(_xy_)        (bufferToImage(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageToWorld(_xy_) (imageToWorld(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldToImage(_xyz_)        (worldToImage(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_worldToImage3(_xyz_)       (_xyz_)
#endif
#define AT_fragColor_data       _bound_fragColor
#define AT_fragColor_bound      1
#define AT_fragColor_stat       ((global IMX_Stat * restrict) _bound_fragColor_stat_void)
#define AT_fragColor_layer      &_bound_fragColor_layer
#define AT_fragColor_border     _bound_fragColor_border
#define AT_fragColor_storage    _bound_fragColor_storage
#define AT_fragColor_channels   _bound_fragColor_channels
#define AT_fragColor_tuplesize  _bound_fragColor_channels
#define AT_fragColor_xres       _bound_fragColor_layer.stat->resolution.x
#define AT_fragColor_yres       _bound_fragColor_layer.stat->resolution.y
#define AT_fragColor_res        convert_float2(_bound_fragColor_layer.stat->resolution)
#define AT_fragColor_set(_val_) _setIndexLinF4(&_bound_fragColor_layer, _bound_idx, _val_, _bound_fragColor_storage, _bound_fragColor_channels)
#define AT_fragColor_setIndex(_xy_, _val_)      _setIndexF4(&_bound_fragColor_layer, _xy_, _val_, _bound_fragColor_storage, _bound_fragColor_channels)
#define AT_fragColor_bufferToImage(_xy_)        (bufferToImage(AT_fragColor_stat, _xy_))
#define AT_fragColor_imageToBuffer(_xy_)        (imageToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_bufferToPixel(_xy_)        (bufferToPixel(AT_fragColor_stat, _xy_))
#define AT_fragColor_pixelToBuffer(_xy_)        (pixelToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_bufferToTexture(_xy_)      (bufferToTexture(AT_fragColor_stat, _xy_))
#define AT_fragColor_textureToBuffer(_xy_)      (textureToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_imageToWorld(_xy_) (imageToWorld(AT_fragColor_stat, _xy_))
#define AT_fragColor_image3ToWorld(_xyz_)       (image3ToWorld(AT_fragColor_stat, _xyz_))
#define AT_fragColor_worldToImage(_xyz_)        (worldToImage(AT_fragColor_stat, _xyz_))
#define AT_fragColor_worldToImage3(_xyz_)       (worldToImage3(AT_fragColor_stat, _xyz_))
#line 1
// HEADER MACROS
// HEADER code from Houdini DIgital Asset (HDA). 
// #bind macros are Houdini Copernicus-specific macros that bind parameters and layers to the OpenCL kernel.
// Contains VEX expressions resolved in HDA UI field before being passed to OpenCL kernel.

// Uniform Shadertoy-like inputs









// Varying Shadertoy-like inputs













// Shadertoy output



// ---- Simplified GLSL Helper Functions ----
// Simple, reliable helper functions for common operations
#include "glslHelpers.h"

// ---- Shadertoy-like texture() for Copernicus ----
#include "textureHelpers.h"


// Shadertoy has global variables that can be called inside functions
// We just initiate empty variables so that code compiles if used inside func()
// They get mapped inside kernel
static float3 iResolution = (float3)(512.0f, 288.0f, 0.0f);
static float iTime = 0.0000f;
static float iTimeDelta = 0.0000f;
static float iFrameRate = 24.0000f;
static int iFrame = 0;
static float4 iMouse = (float4)(0.0000f, 0.0000f, 0.0000f, 0.0000f );
static float4 iDate = (float4)(2025.0000f, 12.0000f, 31.0000f, 60.0000f );
static const float iSampleRate = 44100.0f;

static const IMX_Layer* iChannel0;
static const IMX_Layer* iChannel1;
static const IMX_Layer* iChannel2;
static const IMX_Layer* iChannel3;
static float iChannelTime[4];
static float3 iChannelResolution[4];

// ---- Uniform binding setter (category AG fix + category Q carrier) ----
// Copies the per-pixel bound values into the static-global Shadertoy uniforms.
// DEFINED here in the header (before any transpiled user code), so its body's
// bare uniform tokens (iTime, iResolution, ...) are compiled BEFORE a user
// `#define iTime ...` can reach them. Every value arrives as a PARAMETER — the
// body must never reference @-binding tokens (those map to kernel params).
// Semantics are identical to the old SHADERTOY_INPUTS assignments (same values,
// same order); iTimeDelta/iSampleRate stay untouched, matching the original.
// The final param in_pix_base = (@ix, @iy) is the kernel's pixel base; the
// setter (itself a program-scope fn, so it may call get_global_id()) derives
// the gl_FragCoord offset from it — keeping that arithmetic OUT of the macro.
static void shadertoy_bind_inputs(
    float3 in_iResolution,
    float in_iTime,
    float in_iFrameRate,
    int in_iFrame,
    float4 in_iMouse,
    float4 in_iDate,
    const IMX_Layer* in_iChannel0,
    const IMX_Layer* in_iChannel1,
    const IMX_Layer* in_iChannel2,
    const IMX_Layer* in_iChannel3,
    float3 in_iChannelResolution0,
    float3 in_iChannelResolution1,
    float3 in_iChannelResolution2,
    float3 in_iChannelResolution3,
    int2 in_pix_base)
{
    iResolution = in_iResolution;
    iTime = in_iTime;
    iFrameRate = in_iFrameRate;
    iFrame = in_iFrame;
    iMouse = in_iMouse;
    iDate = in_iDate;
    iChannel0 = in_iChannel0;
    iChannel1 = in_iChannel1;
    iChannel2 = in_iChannel2;
    iChannel3 = in_iChannel3;
    iChannelTime[0] = in_iTime;
    iChannelTime[1] = in_iTime;
    iChannelTime[2] = in_iTime;
    iChannelTime[3] = in_iTime;
    iChannelResolution[0] = in_iChannelResolution0;
    iChannelResolution[1] = in_iChannelResolution1;
    iChannelResolution[2] = in_iChannelResolution2;
    iChannelResolution[3] = in_iChannelResolution3;
    // Category Q carrier: seed the uniform gid->pixel offset DECLARED IN
    // glslHelpers.h (included above; it also defines the GLSL_glFragCoord()
    // accessor helpers call). Benign same-value race: identical value written
    // by every work-item under tilesize==1 (the proven cook geometry, where
    // fragCoord == get_global_id() exactly, so the offset is 0). pixel =
    // get_global_id() + off recovers each work-item's own pixel in ANY
    // function; the seed self-corrects any future *uniform* launch-offset
    // shift. Seeding here (not in transpiler-emitted entry glue) covers every
    // kernel unconditionally, with no transpiler emission required.
    GLSL_glFragCoord_off = in_pix_base - (int2)(get_global_id(0), get_global_id(1));
}


#ifdef CUBEMAP_RENDERPASS
    // DO_CUBEMAP only mentions shadertoy_cubemap_bind, rayDir and @-tokens —
    // the `&iResolution` token is hidden inside the header-defined wrapper so a
    // user `#define iResolution ...` cannot poison it.
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap_bind(AT_ix,AT_iy,AT_xres,AT_yres,&rayDir);
#else
    #define DO_CUBEMAP /* nothing */
#endif

// SHADERTOY_INPUTS opens every kernel body. It no longer contains any bare
// Shadertoy uniform name-token: all uniform assignments are delegated to
// shadertoy_bind_inputs() (defined above, before user #defines). Only the
// kernel-scope locals fragCoord/fragColor (read by the transpiled glue) and
// @-binding tokens remain — neither is poisonable by a user `#define iTime`.
// The trailing (int2)(@ix, @iy) hands the pixel base to the setter so it can
// derive the uniform gl_FragCoord offset (category Q enabler).
#define SHADERTOY_INPUTS \
    shadertoy_bind_inputs( \
        (float3)(AT_xres, AT_yres, 0.0f), \
        AT_Time, \
        AT_iFrameRate, \
        AT_iFrame, \
        AT_iMouse, \
        AT_iDate, \
        AT_iChannel0_layer, \
        AT_iChannel1_layer, \
        AT_iChannel2_layer, \
        AT_iChannel3_layer, \
        (float3)(AT_iChannel0_res, 0.0f), \
        (float3)(AT_iChannel1_res, 0.0f), \
        (float3)(AT_iChannel2_res, 0.0f), \
        (float3)(AT_iChannel3_res, 0.0f), \
        (int2)(AT_ix, AT_iy)); \
    float2 fragCoord = AT_fragCoord; \
    if (!AT_fragCoord_bound) { fragCoord = (float2)(AT_ix, AT_iy); }\
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP

// mainCubemap renderpass helper    
// Unpacks 3x2 cubemap layout to ray direction and adjusts resolution to cube face
// Standard cubemap layout:
//   [+X][-X][+Z]
//   [+Y][-Y][-Z]
static void shadertoy_cubemap(int ix, int iy, int xres, int yres, 
                              float3* rayDir, float3* iResolution)
{
    // Calculate individual face dimensions
    int face_width = xres / 3;
    int face_height = yres / 2;
    
    // Update iResolution to single face size
    *iResolution = (float3)(face_width, face_height, 0.0f);
    
    // Determine which face we're rendering (0-2 for x, 0-1 for y)
    int face_x = ix / face_width;
    int face_y = iy / face_height;
    
    // Calculate local UV coordinates within the face (-1 to 1 range)
    float2 local_uv = (float2)(
        (float)(ix % face_width) / (float)face_width * 2.0f - 1.0f,
        (float)(iy % face_height) / (float)face_height * 2.0f - 1.0f
    );
    
    // Map face position to ray direction
    // Each face represents a direction in the cube
    if (face_x == 0 && face_y == 0) {
        // +X face (right)
        *rayDir = (float3)(1.0f, -local_uv.y, -local_uv.x);
    } 
    else if (face_x == 1 && face_y == 0) {
        // -X face (left)
        *rayDir = (float3)(-1.0f, -local_uv.y, local_uv.x);
    } 
    else if (face_x == 0 && face_y == 1) {
        // +Y face (up)
        *rayDir = (float3)(local_uv.x, 1.0f, local_uv.y);
    } 
    else if (face_x == 1 && face_y == 1) {
        // -Y face (down)
        *rayDir = (float3)(local_uv.x, -1.0f, -local_uv.y);
    }
    else if (face_x == 2 && face_y == 0) {
        // +Z face (forward)
        *rayDir = (float3)(local_uv.x, -local_uv.y, 1.0f);
    } 
    else if (face_x == 2 && face_y == 1) {
        // -Z face (back)
        *rayDir = (float3)(-local_uv.x, -local_uv.y, -1.0f);
    }
}

// Cubemap binding wrapper (category AG fix): keeps the `&iResolution` token
// OUT of the DO_CUBEMAP macro body so a user `#define iResolution ...` cannot
// poison it. Defined here (before user code); DO_CUBEMAP only mentions this
// name, `rayDir`, and @-binding tokens. Body forwards to shadertoy_cubemap().
static void shadertoy_cubemap_bind(int ix, int iy, int xres, int yres,
                                   float3* rayDir)
{
    shadertoy_cubemap(ix, iy, xres, yres, rayDir, &iResolution);
}