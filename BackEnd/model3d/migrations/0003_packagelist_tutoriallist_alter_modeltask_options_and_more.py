# Generated by Django 5.0.6 on 2024-07-17 07:36

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model3d', '0002_rename_image_url_modeltask_image_download_url_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='PackageList',
            fields=[
                ('pid', models.AutoField(primary_key=True, serialize=False, verbose_name='套装_id')),
                ('caption', models.CharField(max_length=64, verbose_name='套装名称')),
                ('description', models.CharField(blank=True, max_length=64, null=True, verbose_name='套装描述')),
                ('createTime_Internal', models.DateTimeField(auto_now_add=True, verbose_name='创建日期')),
                ('alterTime_Internal', models.DateTimeField(auto_now=True, verbose_name='修改日期')),
            ],
            options={
                'verbose_name': '元件套装表',
                'verbose_name_plural': '元件套装表',
            },
        ),
        migrations.CreateModel(
            name='TutorialList',
            fields=[
                ('pid', models.AutoField(primary_key=True, serialize=False, verbose_name='教程_id')),
                ('title', models.CharField(max_length=64, verbose_name='教程标题')),
                ('description', models.CharField(blank=True, max_length=64, null=True, verbose_name='教程描述')),
                ('componentSum', models.IntegerField(default=0, verbose_name='含有元件数')),
                ('imgFile', models.FileField(blank=True, null=True, upload_to='tutotial/', verbose_name='教程img文件')),
                ('key', models.CharField(blank=True, max_length=256, null=True, verbose_name='教程key')),
                ('createTime_Internal', models.DateTimeField(auto_now_add=True, verbose_name='创建日期')),
                ('alterTime_Internal', models.DateTimeField(auto_now=True, verbose_name='修改日期')),
            ],
            options={
                'verbose_name': '生成教程表',
                'verbose_name_plural': '生成教程表',
            },
        ),
        migrations.AlterModelOptions(
            name='modeltask',
            options={'verbose_name': '生成任务表', 'verbose_name_plural': '生成任务表'},
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='bricks',
            field=models.TextField(verbose_name='选择的元件'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='created_time',
            field=models.DateTimeField(auto_now_add=True, verbose_name='创建日期'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='image_download_url',
            field=models.URLField(verbose_name='模型预览图保存地址'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='lego_url',
            field=models.URLField(verbose_name='乐高方案保存地址'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='model_download_url',
            field=models.URLField(verbose_name='模型保存地址'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='prompt',
            field=models.TextField(verbose_name='提示词'),
        ),
        migrations.AlterField(
            model_name='modeltask',
            name='task_id',
            field=models.CharField(max_length=255, unique=True, verbose_name='任务_id'),
        ),
        migrations.CreateModel(
            name='ComponentList',
            fields=[
                ('pid', models.AutoField(primary_key=True, serialize=False, verbose_name='元件_id')),
                ('sn', models.CharField(max_length=64, verbose_name='元件编号')),
                ('caption', models.CharField(max_length=64, verbose_name='元件名称')),
                ('btype', models.IntegerField(choices=[(0, '未定义'), (1, '条'), (2, '板'), (3, '块'), (4, '其他')], default=0, verbose_name='元件类型')),
                ('bcolor', models.IntegerField(choices=[(1, '白'), (2, '黑'), (3, '红'), (4, '绿')], default=0, verbose_name='元件颜色')),
                ('imgFile', models.FileField(blank=True, null=True, upload_to='bricks/', verbose_name='元件img文件')),
                ('createTime_Internal', models.DateTimeField(auto_now_add=True, verbose_name='创建日期')),
                ('alterTime_Internal', models.DateTimeField(auto_now=True, verbose_name='修改日期')),
                ('package', models.ForeignKey(default=1, on_delete=django.db.models.deletion.PROTECT, to='model3d.packagelist', verbose_name='所属套装')),
            ],
            options={
                'verbose_name': '元件表',
                'verbose_name_plural': '元件表',
            },
        ),
    ]
