# Note: plugindir is set in configure

plugin_LTLIBRARIES = libgstmxnet.la

# sources used to compile this plug-in
libgstmxnet_la_SOURCES = gstmxnet_cpp.cc gstmxnet.c gstmxnet.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstmxnet_la_CFLAGS = $(GST_CFLAGS)
libgstmxnet_la_CPPFLAGS = $(GST_CFLAGS)
libgstmxnet_la_LIBADD = $(GST_LIBS)
libgstmxnet_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstmxnet_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = gstmxnet.h
