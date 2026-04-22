GPU := H100

SRC := tk_evoattention.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_tk_evoattention.py
CONFIG := pytorch
NVCCFLAGS := -DTORCH_COMPILE

include ../../common.mk
